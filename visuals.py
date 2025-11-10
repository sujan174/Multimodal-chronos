import yfinance as yf
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from tqdm import tqdm
import pywt
from PIL import Image
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "JPM", "BAC", "GS", "MS", "C", "WFC", "V", "MA",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "CVS",
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "DIS",
    "XOM", "CVX", "COP", "SLB", "EOG",
]

TRAIN_START = "2022-01-01"
VAL_START = "2024-01-01"
TEST_START = "2024-07-01"
TEST_END = "2024-12-31"

LOOKBACK_WINDOW = 60
PREDICTION_HORIZON = 1
WAVELET_SCALES = np.arange(1, 32)
WAVELETS = ['morl', 'cmor', 'gaus1']

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 1e-3
BACKBONE_LR_RATIO = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
LABEL_SMOOTHING = 0.1

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15
EARLY_STOP_PATIENCE = 5
GRAD_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

if DEVICE == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

print(f"Using device: {DEVICE} | AMP: {USE_AMP}")

def fetch_prices(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty or len(df) < LOOKBACK_WINDOW + 10:
            return None

        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index = df.index.tz_localize(None)

        return df
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None

def calculate_technical_features(df):
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['volatility'] = df['return_1d'].rolling(20).std()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])

    return df

def generate_wavelet_scalograms(prices_df, output_path, ticker):
    ticker_path = os.path.join(output_path, ticker)

    if os.path.exists(ticker_path):
        return

    os.makedirs(ticker_path, exist_ok=True)

    price_series = prices_df['close']
    dates = prices_df.index

    for i in range(LOOKBACK_WINDOW, len(price_series)):
        current_date = dates[i]
        price_window = price_series.iloc[i-LOOKBACK_WINDOW:i]

        normalized = (price_window - price_window.mean()) / (price_window.std() + 1e-9)

        channels = []
        for wavelet in WAVELETS:
            coeffs, _ = pywt.cwt(normalized, WAVELET_SCALES, wavelet)
            channel = np.log1p(np.abs(coeffs))

            channel_min, channel_max = channel.min(), channel.max()
            if channel_max > channel_min:
                channel = ((channel - channel_min) / (channel_max - channel_min) * 255).astype(np.uint8)
            else:
                channel = np.zeros_like(channel, dtype=np.uint8)

            channels.append(Image.fromarray(channel))

        img = Image.merge('RGB', channels)
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(os.path.join(ticker_path, f"{current_date.strftime('%Y-%m-%d')}.png"))

class WaveletDataset(Dataset):
    def __init__(self, image_paths, targets, tickers, transform):
        self.image_paths = image_paths
        self.targets = targets
        self.tickers = tickers
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.targets[idx]
        return self.transform(image), torch.tensor([target], dtype=torch.float32)

def get_transforms(augment=True):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class EfficientNetPredictor(nn.Module):
    def __init__(self, dropout=DROPOUT, freeze_backbone=True):
        super().__init__()

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            print("Backbone frozen for Phase 1")

        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def unfreeze_backbone(self, progressive=True, num_layers=None):
        if progressive and num_layers:
            layers = list(self.backbone.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfrozen last {num_layers} backbone layers")
        else:
            for param in self.backbone.features.parameters():
                param.requires_grad = True
            print("Entire backbone unfrozen")

    def forward(self, x):
        return self.backbone(x)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

class MetricsTracker:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rates': []
        }

    def update(self, metrics):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

def evaluate_model(model, loader, criterion, device, split_name="VAL"):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    preds = np.array(all_preds)
    probs = np.array(all_probs)
    targets = np.array(all_targets)

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(targets, preds) * 100,
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1_score": f1_score(targets, preds, zero_division=0),
        "auc_roc": roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0
    }

    cm = confusion_matrix(targets, preds)

    print(f"\n{split_name} Loss: {metrics['loss']:.4f}, "
          f"Acc: {metrics['accuracy']:.2f}%, "
          f"Prec: {metrics['precision']:.3f}, "
          f"Rec: {metrics['recall']:.3f}, "
          f"F1: {metrics['f1_score']:.3f}, "
          f"AUC: {metrics['auc_roc']:.3f}")
    print(f"{split_name} Confusion Matrix:\n{cm}")

    return metrics

def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                epoch, total_epochs, scaler=None, phase=1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Phase {phase} | Epoch {epoch+1}/{total_epochs}")

    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets) / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix({
            'loss': loss.item() * GRAD_ACCUM_STEPS,
            'lr': optimizer.param_groups[0]['lr']
        })

        if device == "mps" and batch_idx % 50 == 0:
            torch.mps.empty_cache()

    avg_loss = epoch_loss / len(loader)
    return avg_loss

def train_model(train_loader, val_loader, test_loader, device, class_weights=None):
    print("\nTraining Configuration")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} | Grad Accumulation: {GRAD_ACCUM_STEPS}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Phase 1 Epochs: {PHASE1_EPOCHS} | Phase 2 Epochs: {PHASE2_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE} | Backbone LR Ratio: {BACKBONE_LR_RATIO}")

    model = EfficientNetPredictor(freeze_backbone=True).to(device)
    print(f"Trainable parameters: {model.count_trainable_params():,}")

    if class_weights is not None:
        pos_weight_value = float(class_weights[1] / class_weights[0])
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
        )
        print(f"Using class weights: {class_weights} | Pos weight: {pos_weight_value:.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    tracker = MetricsTracker()
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')
    best_val_f1 = 0.0

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    print("\nPhase 1: Warmup - Training Classifier Head (Frozen Backbone)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader) // GRAD_ACCUM_STEPS,
        epochs=PHASE1_EPOCHS,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    for epoch in range(PHASE1_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, PHASE1_EPOCHS, scaler, phase=1
        )

        print(f"\nPhase 1 Epoch {epoch+1}: Train Loss: {train_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, criterion, device, "VALIDATION")

        tracker.update({
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score'],
            'val_auc': val_metrics['auc_roc'],
            'learning_rates': optimizer.param_groups[0]['lr']
        })

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'phase': 1
            }, './models/multi_stock_phase1_best.pth')
            print(f"Phase 1 checkpoint saved (F1: {best_val_f1:.3f})")

        if device == "mps":
            torch.mps.empty_cache()

    print("\nPhase 2: Fine-Tuning - Full Model Training (Unfrozen Backbone)")

    model.unfreeze_backbone(progressive=False)
    print(f"Trainable parameters: {model.count_trainable_params():,}")

    optimizer = optim.AdamW([
        {
            'params': model.backbone.features.parameters(),
            'lr': LEARNING_RATE * BACKBONE_LR_RATIO,
            'weight_decay': WEIGHT_DECAY
        },
        {
            'params': model.backbone.classifier.parameters(),
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        }
    ], betas=(0.9, 0.999))

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LEARNING_RATE * BACKBONE_LR_RATIO, LEARNING_RATE],
        steps_per_epoch=len(train_loader) // GRAD_ACCUM_STEPS,
        epochs=PHASE2_EPOCHS,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode='max')

    for epoch in range(PHASE2_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, PHASE2_EPOCHS, scaler, phase=2
        )

        print(f"\nPhase 2 Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
        print(f"LR Backbone: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Classifier: {optimizer.param_groups[1]['lr']:.6f}")

        val_metrics = evaluate_model(model, val_loader, criterion, device, "VALIDATION")

        tracker.update({
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score'],
            'val_auc': val_metrics['auc_roc'],
            'learning_rates': optimizer.param_groups[1]['lr']
        })

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'phase': 2
            }, './models/multi_stock_best.pth')
            print(f"Best model saved (F1: {best_val_f1:.3f})")

        early_stopping(val_metrics['f1_score'])
        if early_stopping.early_stop:
            print(f"\nEarly stop triggered at epoch {epoch+1}")
            break

        if device == "mps":
            torch.mps.empty_cache()

    tracker.save('./models/training_history.json')

    print("\nFinal Evaluation on Test Set")

    checkpoint = torch.load('./models/multi_stock_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']} (F1: {checkpoint['val_f1']:.3f})")

    val_metrics = evaluate_model(model, val_loader, criterion, device, "VALIDATION")
    test_metrics = evaluate_model(model, test_loader, criterion, device, "TEST")

    return model, val_metrics, test_metrics, tracker

def main():
    print("Production-Grade Multi-Stock Prediction System")
    print("Fine-tuned EfficientNet with Best ML Practices")
    print(f"Tickers: {len(TICKERS)} stocks")
    print(f"Training Period: {TRAIN_START} to {TEST_END}")
    print(f"Device: {DEVICE} | Mixed Precision: {USE_AMP}")

    os.makedirs('./data', exist_ok=True)
    os.makedirs('./features', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    scalogram_base = './features/multi_stock_scalograms'
    os.makedirs(scalogram_base, exist_ok=True)

    print("\nFetching and processing stock data...")

    all_train_data = []
    all_val_data = []
    all_test_data = []
    successful_tickers = []

    for ticker in tqdm(TICKERS, desc="Processing Tickers"):
        df = fetch_prices(ticker, TRAIN_START, TEST_END)
        if df is None:
            continue

        df = calculate_technical_features(df)
        df['target'] = (df['close'].shift(-PREDICTION_HORIZON) > df['close']).astype(int)
        df.dropna(inplace=True)

        if len(df) < LOOKBACK_WINDOW + 10:
            continue

        generate_wavelet_scalograms(df, scalogram_base, ticker)

        train_df = df.loc[df.index < VAL_START].copy()
        val_df = df.loc[(df.index >= VAL_START) & (df.index < TEST_START)].copy()
        test_df = df.loc[df.index >= TEST_START].copy()

        for split_df in [train_df, val_df, test_df]:
            split_df['ticker'] = ticker
            split_df['image_path'] = split_df.index.strftime('%Y-%m-%d').map(
                lambda x: os.path.join(scalogram_base, ticker, f"{x}.png")
            )

        train_df = train_df[train_df['image_path'].apply(os.path.exists)]
        val_df = val_df[val_df['image_path'].apply(os.path.exists)]
        test_df = test_df[test_df['image_path'].apply(os.path.exists)]

        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            all_train_data.append(train_df)
            all_val_data.append(val_df)
            all_test_data.append(test_df)
            successful_tickers.append(ticker)

    print(f"\nProcessed {len(successful_tickers)}/{len(TICKERS)} tickers")
    print(f"Tickers: {', '.join(successful_tickers[:10])}...")

    train_combined = pd.concat(all_train_data, ignore_index=False)
    val_combined = pd.concat(all_val_data, ignore_index=False)
    test_combined = pd.concat(all_test_data, ignore_index=False)

    print(f"\nDataset Statistics")
    print(f"Train: {len(train_combined)} samples")
    print(f"Validation: {len(val_combined)} samples")
    print(f"Test: {len(test_combined)} samples")
    print(f"\nClass Distribution")
    print(f"Train - Positive: {train_combined['target'].sum()} ({train_combined['target'].mean()*100:.1f}%)")
    print(f"Val   - Positive: {val_combined['target'].sum()} ({val_combined['target'].mean()*100:.1f}%)")
    print(f"Test  - Positive: {test_combined['target'].sum()} ({test_combined['target'].mean()*100:.1f}%)")

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_combined['target']),
        y=train_combined['target'].values
    )
    print(f"\nClass Weights: Negative: {class_weights[0]:.3f} | Positive: {class_weights[1]:.3f}")

    print("\nCreating datasets and dataloaders...")

    train_dataset = WaveletDataset(
        train_combined['image_path'].values,
        train_combined['target'].values,
        train_combined['ticker'].values,
        get_transforms(augment=True)
    )

    val_dataset = WaveletDataset(
        val_combined['image_path'].values,
        val_combined['target'].values,
        val_combined['ticker'].values,
        get_transforms(augment=False)
    )

    test_dataset = WaveletDataset(
        test_combined['image_path'].values,
        test_combined['target'].values,
        test_combined['ticker'].values,
        get_transforms(augment=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Loaders: Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    model, val_metrics, test_metrics, tracker = train_model(
        train_loader, val_loader, test_loader, DEVICE, class_weights
    )

    results = {
        "configuration": {
            "tickers": successful_tickers,
            "num_tickers": len(successful_tickers),
            "train_samples": len(train_combined),
            "val_samples": len(val_combined),
            "test_samples": len(test_combined),
            "lookback_window": LOOKBACK_WINDOW,
            "prediction_horizon": PREDICTION_HORIZON,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "backbone_lr_ratio": BACKBONE_LR_RATIO,
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "device": DEVICE,
            "mixed_precision": USE_AMP
        },
        "class_distribution": {
            "train_positive_pct": float(train_combined['target'].mean() * 100),
            "val_positive_pct": float(val_combined['target'].mean() * 100),
            "test_positive_pct": float(test_combined['target'].mean() * 100),
            "class_weights": class_weights.tolist()
        },
        "validation_metrics": {
            "loss": float(val_metrics['loss']),
            "accuracy": float(val_metrics['accuracy']),
            "precision": float(val_metrics['precision']),
            "recall": float(val_metrics['recall']),
            "f1_score": float(val_metrics['f1_score']),
            "auc_roc": float(val_metrics['auc_roc'])
        },
        "test_metrics": {
            "loss": float(test_metrics['loss']),
            "accuracy": float(test_metrics['accuracy']),
            "precision": float(test_metrics['precision']),
            "recall": float(test_metrics['recall']),
            "f1_score": float(test_metrics['f1_score']),
            "auc_roc": float(test_metrics['auc_roc'])
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('./models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTraining Complete - Summary")
    print(f"Model saved: ./models/multi_stock_best.pth")
    print(f"Results saved: ./models/training_results.json")
    print(f"Training history: ./models/training_history.json")
    print(f"Tickers trained: {len(successful_tickers)}")
    print(f"\nValidation Performance")
    print(f"  Accuracy:  {val_metrics['accuracy']:.2f}%")
    print(f"  Precision: {val_metrics['precision']:.3f}")
    print(f"  Recall:    {val_metrics['recall']:.3f}")
    print(f"  F1-Score:  {val_metrics['f1_score']:.3f}")
    print(f"  AUC-ROC:   {val_metrics['auc_roc']:.3f}")
    print(f"\nTest Performance")
    print(f"  Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.3f}")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']:.3f}")

    print("\nInference Example")
    print("To use the trained model for predictions:")
    print("""
    model = EfficientNetPredictor(freeze_backbone=False)
    checkpoint = torch.load('./models/multi_stock_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        image = Image.open('path/to/scalogram.png').convert('RGB')
        transform = get_transforms(augment=False)
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        logit = model(image_tensor)
        probability = torch.sigmoid(logit).item()
        prediction = 1 if probability > 0.5 else 0

        print(f"Prediction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"Confidence: {probability:.2%}")
    """)

    return model, results

def plot_training_history(history_path='./models/training_history.json'):
    try:
        import matplotlib.pyplot as plt

        with open(history_path, 'r') as f:
            history = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(history['val_f1'], label='Val F1', color='orange')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='red')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('./models/training_history.png', dpi=300, bbox_inches='tight')
        print(f"Training history saved to ./models/training_history.png")

    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
    except Exception as e:
        print(f"Failed to plot training history: {e}")

def analyze_per_ticker_performance(model, test_loader, test_combined, device):
    model.eval()
    ticker_predictions = {}

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            batch_start = idx * test_loader.batch_size
            batch_end = min(batch_start + len(images), len(test_combined))

            for i, ticker in enumerate(test_combined.iloc[batch_start:batch_end]['ticker']):
                if ticker not in ticker_predictions:
                    ticker_predictions[ticker] = {'preds': [], 'targets': [], 'probs': []}

                ticker_predictions[ticker]['preds'].append(preds[i])
                ticker_predictions[ticker]['targets'].append(targets[i].item())
                ticker_predictions[ticker]['probs'].append(probs[i])

    ticker_metrics = {}
    for ticker, data in ticker_predictions.items():
        preds = np.array(data['preds'])
        targets = np.array(data['targets'])
        probs = np.array(data['probs'])

        ticker_metrics[ticker] = {
            'accuracy': accuracy_score(targets, preds) * 100,
            'f1_score': f1_score(targets, preds, zero_division=0),
            'auc_roc': roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0,
            'num_samples': len(preds)
        }

    sorted_tickers = sorted(ticker_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)

    print("\nPer-Ticker Performance - Top 10 Best:")
    for ticker, metrics in sorted_tickers[:10]:
        print(f"  {ticker:6s} | Acc: {metrics['accuracy']:5.2f}% | "
              f"F1: {metrics['f1_score']:.3f} | AUC: {metrics['auc_roc']:.3f} | "
              f"Samples: {metrics['num_samples']}")

    return ticker_metrics

if __name__ == "__main__":
    model, results = main()

    try:
        plot_training_history()
    except Exception as e:
        print(f"Skipping plot generation: {e}")

    print("\nAll tasks finished successfully!")
