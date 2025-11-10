import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import pywt
from PIL import Image
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from chronos import ChronosPipeline
    from peft import LoraConfig, get_peft_model, TaskType
    CHRONOS_AVAILABLE = True
    print("Chronos with LoRA available")
except ImportError:
    CHRONOS_AVAILABLE = False
    print("Chronos unavailable - using LSTM fallback")

TRAIN_START = "2022-06-01"
VAL_START = "2024-01-01"
TEST_START = "2024-07-01"
TEST_END = "2024-12-31"

TRAIN_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA",
    "JPM", "JNJ", "V", "WMT", "PG", "XOM"
]

EFFICIENTNET_MODEL_PATH = './models/multi_stock_best.pth'
CHRONOS_MODEL_NAME = "amazon/chronos-t5-tiny"
SCALOGRAM_PATH = './features/hybrid_scalograms'
MODEL_SAVE_PATH = './models/hybrid_fusion_model.pth'
RESULTS_PATH = './hybrid_results'

LOOKBACK_WINDOW = 30
PREDICTION_HORIZON = 1
WAVELET_SCALES = np.arange(1, 32)
WAVELETS = ['morl', 'cmor', 'gaus1']
IMAGE_SIZE = (224, 224)
CHRONOS_CONTEXT_LENGTH = 256
CHRONOS_PREDICTION_LENGTH = 1

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 5

EMBEDDING_DIM = 128
NUM_ATTENTION_HEADS = 4
NUM_FUSION_LAYERS = 2
TECHNICAL_FEATURE_DIM = 32

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

def print_memory_usage(device, message=""):
    if device == "mps" and message:
        print(message)
    elif device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{message} | GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif message:
        print(message)

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df):
        df = df.copy()

        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        df['parkinson_vol'] = TechnicalIndicators._parkinson_volatility(df)

        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']

        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(float)
        df['sma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(float)
        df['ema_cross_5_20'] = (df['ema_5'] > df['ema_20']).astype(float)

        df['rsi_14'] = TechnicalIndicators._calculate_rsi(df['close'], 14)
        df['rsi_7'] = TechnicalIndicators._calculate_rsi(df['close'], 7)
        df['rsi_21'] = TechnicalIndicators._calculate_rsi(df['close'], 21)

        df['macd'], df['macd_signal'], df['macd_histogram'] = TechnicalIndicators._calculate_macd(df['close'])

        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        df['stoch_k'], df['stoch_d'] = TechnicalIndicators._calculate_stochastic(df)

        df['atr_14'] = TechnicalIndicators._calculate_atr(df, 14)
        df['atr_7'] = TechnicalIndicators._calculate_atr(df, 7)

        df['adx_14'] = TechnicalIndicators._calculate_adx(df, 14)

        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
        df['volume_std'] = df['volume'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / (df['volume_std'] + 1e-10)

        df['obv'] = TechnicalIndicators._calculate_obv(df)
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

        df['mfi'] = TechnicalIndicators._calculate_mfi(df, 14)

        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc_10'] = df['close'].pct_change(10) * 100

        df['hl_ratio'] = df['high'] / df['low']
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()

        df['williams_r'] = TechnicalIndicators._calculate_williams_r(df, 14)

        df['cci'] = TechnicalIndicators._calculate_cci(df, 20)

        df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

        df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                             (df['high'].shift(1) > df['high'].shift(2))).astype(float)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                           (df['low'].shift(1) < df['low'].shift(2))).astype(float)

        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(float)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(float)

        return df

    @staticmethod
    def _calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    @staticmethod
    def _calculate_stochastic(df, period=14):
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(3).mean()
        return k, d

    @staticmethod
    def _calculate_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _calculate_adx(df, period):
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = TechnicalIndicators._calculate_atr(df, 1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx

    @staticmethod
    def _calculate_obv(df):
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def _calculate_mfi(df, period):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi

    @staticmethod
    def _calculate_williams_r(df, period):
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)

    @staticmethod
    def _calculate_cci(df, period):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma_tp) / (0.015 * mad + 1e-10)

    @staticmethod
    def _parkinson_volatility(df, window=20):
        return np.sqrt(1 / (4 * np.log(2)) *
                      (np.log(df['high'] / df['low']) ** 2).rolling(window).mean())

class EfficientNetPredictor(nn.Module):
    def __init__(self, dropout=0.4, freeze_backbone=False):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)

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

    def forward(self, x):
        return self.backbone(x)

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()

        full_model = EfficientNetPredictor()
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        full_model.load_state_dict(checkpoint['model_state_dict'])

        self.backbone = full_model.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = full_model.backbone.classifier[0]
        self.fc1 = full_model.backbone.classifier[1]
        self.bn1 = full_model.backbone.classifier[2]
        self.gelu1 = full_model.backbone.classifier[3]
        self.dropout2 = full_model.backbone.classifier[4]
        self.fc2 = full_model.backbone.classifier[5]
        self.bn2 = full_model.backbone.classifier[6]

        for param in self.parameters():
            param.requires_grad = False

        print("EfficientNet features frozen")

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x

class LSTMTimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, num_layers=2, embedding_dim=512, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        print(f"LSTM encoder initialized (output: {embedding_dim})")

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        lstm_out, (hidden, cell) = self.lstm(x)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        embeddings = self.projection(combined)
        confidence = torch.ones(x.size(0), device=x.device)

        return embeddings, confidence

class LoRAChronosEncoder(nn.Module):
    def __init__(self, model_name="amazon/chronos-t5-tiny", device="cpu", lora_rank=8, lora_alpha=16):
        super().__init__()

        self.device = device
        self.context_length = CHRONOS_CONTEXT_LENGTH

        print(f"Loading {model_name}...")

        target_device = "cpu" if device == "mps" else device
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=target_device,
            torch_dtype=torch.float32
        )

        self.model = self.pipeline.model

        if hasattr(self.model, 'config') and not hasattr(self.model.config, 'get'):
            def config_get(key, default=None):
                return getattr(self.model.config, key, default)
            self.model.config.get = config_get

        if hasattr(self.model, 'config'):
            self.hidden_size = getattr(self.model.config, 'd_model', 256)
        else:
            self.hidden_size = 256

        print(f"Hidden size: {self.hidden_size}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        self.embedding_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        print(f"LoRA-Chronos ready on {target_device}")

    def encode_time_series(self, time_series):
        batch_size = time_series.size(0)
        all_hidden = []

        if hasattr(self.model, 'base_model'):
            base_model = self.model.base_model.model
        else:
            base_model = self.model

        for i in range(batch_size):
            series = time_series[i].cpu().numpy()

            if len(series) < self.context_length:
                series = np.pad(series, (self.context_length - len(series), 0), mode='edge')
            elif len(series) > self.context_length:
                series = series[-self.context_length:]

            context = torch.tensor(series, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.enable_grad():
                try:
                    scaled_context = context / (context.abs().max() + 1e-10)
                    num_bins = 1024
                    scaled_bins = ((scaled_context + 1) / 2 * (num_bins - 1)).long()
                    scaled_bins = torch.clamp(scaled_bins, 0, num_bins - 1)

                    if hasattr(base_model, 'encoder'):
                        encoder_outputs = base_model.encoder(
                            input_ids=scaled_bins,
                            return_dict=True
                        )
                    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'encoder'):
                        encoder_outputs = base_model.model.encoder(
                            input_ids=scaled_bins,
                            return_dict=True
                        )
                    else:
                        raise AttributeError(f"Cannot find encoder")

                    hidden = encoder_outputs.last_hidden_state.mean(dim=1)
                    all_hidden.append(hidden)

                except Exception as e:
                    print(f"Encoding failed: {e}")
                    all_hidden.append(torch.zeros(1, self.hidden_size, device=self.device))

        return torch.cat(all_hidden, dim=0)

    def forward(self, x):
        hidden_states = self.encode_time_series(x)
        embeddings = self.embedding_projection(hidden_states)
        confidence = self.confidence_head(hidden_states.detach()).squeeze(-1)
        return embeddings, confidence

class ChronosFeatureExtractor:
    def __init__(self, model_name=CHRONOS_MODEL_NAME, device=DEVICE, use_lora=True):
        self.device = device
        self.context_length = CHRONOS_CONTEXT_LENGTH
        self.use_chronos = CHRONOS_AVAILABLE and use_lora

        if self.use_chronos:
            print("Using LoRA-Chronos encoder")
            self.encoder = LoRAChronosEncoder(
                model_name=model_name,
                device=device,
                lora_rank=8,
                lora_alpha=16
            ).to(device)
        else:
            print("Using LSTM encoder")
            self.encoder = LSTMTimeSeriesEncoder(
                input_dim=1,
                hidden_dim=256,
                num_layers=2,
                embedding_dim=512,
                dropout=0.2
            ).to(device)

    def extract_embeddings(self, time_series_batch, return_confidence=False):
        processed_series = []

        for series in time_series_batch:
            if len(series) < self.context_length:
                series = np.pad(series, (self.context_length - len(series), 0), mode='edge')
            elif len(series) > self.context_length:
                series = series[-self.context_length:]

            processed_series.append(series)

        batch_tensor = torch.tensor(
            np.stack(processed_series),
            dtype=torch.float32
        ).to(self.device)

        embeddings, confidence = self.encoder(batch_tensor)

        if return_confidence:
            return embeddings, confidence
        else:
            return embeddings

class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, timeseries_dim, technical_dim,
                 embed_dim=EMBEDDING_DIM, num_heads=NUM_ATTENTION_HEADS, dropout=DROPOUT):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.timeseries_proj = nn.Linear(timeseries_dim, embed_dim)
        self.technical_proj = nn.Linear(technical_dim, embed_dim)

        self.visual_to_ts_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ts_to_visual_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.technical_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, visual_features, timeseries_features, technical_features):
        visual_emb = self.visual_proj(visual_features).unsqueeze(1)
        ts_emb = self.timeseries_proj(timeseries_features).unsqueeze(1)
        tech_emb = self.technical_proj(technical_features).unsqueeze(1)

        visual_attended, _ = self.visual_to_ts_attn(visual_emb, ts_emb, ts_emb)
        visual_attended = self.norm1(visual_attended + visual_emb)

        ts_attended, _ = self.ts_to_visual_attn(ts_emb, visual_emb, visual_emb)
        ts_attended = self.norm2(ts_attended + ts_emb)

        tech_attended, _ = self.technical_self_attn(tech_emb, tech_emb, tech_emb)
        tech_attended = self.norm3(tech_attended + tech_emb)

        visual_out = visual_attended + self.ffn(visual_attended)
        ts_out = ts_attended + self.ffn(ts_attended)
        tech_out = tech_attended + self.ffn(tech_attended)

        combined = torch.cat([
            visual_out.squeeze(1),
            ts_out.squeeze(1),
            tech_out.squeeze(1)
        ], dim=1)

        fused = self.fusion(combined)

        return fused

class HybridFusionModel(nn.Module):
    def __init__(self, efficientnet_path, chronos_model_name, num_technical_features):
        super().__init__()

        self.visual_extractor = EfficientNetFeatureExtractor(efficientnet_path)
        self.chronos_extractor = ChronosFeatureExtractor(chronos_model_name)

        self.technical_encoder = nn.Sequential(
            nn.Linear(num_technical_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, TECHNICAL_FEATURE_DIM),
            nn.LayerNorm(TECHNICAL_FEATURE_DIM),
            nn.GELU()
        )

        visual_dim = 128
        timeseries_dim = 512

        print(f"Embedding dims - TS: {timeseries_dim}, Visual: {visual_dim}, Technical: {TECHNICAL_FEATURE_DIM}")

        self.fusion_module = CrossAttentionFusion(
            visual_dim=visual_dim,
            timeseries_dim=timeseries_dim,
            technical_dim=TECHNICAL_FEATURE_DIM,
            embed_dim=EMBEDDING_DIM,
            num_heads=NUM_ATTENTION_HEADS,
            dropout=DROPOUT
        )

        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
            nn.Linear(64, 1)
        )

        self.modality_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, images, time_series, technical_features, return_confidence=False):
        visual_features = self.visual_extractor(images)

        ts_embeddings, ts_confidence = self.chronos_extractor.extract_embeddings(
            time_series.cpu().numpy(),
            return_confidence=True
        )
        ts_embeddings = ts_embeddings.to(images.device)
        ts_confidence = ts_confidence.to(images.device)

        technical_encoded = self.technical_encoder(technical_features)

        fused_features = self.fusion_module(
            visual_features,
            ts_embeddings,
            technical_encoded
        )

        logits = self.classifier(fused_features)

        if return_confidence:
            return logits, ts_confidence
        else:
            return logits

    def get_modality_weights(self):
        weights = F.softmax(self.modality_weights, dim=0)
        return {
            'visual': weights[0].item(),
            'timeseries': weights[1].item(),
            'technical': weights[2].item()
        }

class HybridMultiModalDataset(Dataset):
    def __init__(self, df, scalogram_path, transform, scaler=None, fit_scaler=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.scalogram_path = scalogram_path

        exclude_cols = ['target', 'ticker', 'image_path', 'open', 'high', 'low', 'close', 'volume']
        self.technical_cols = [col for col in df.columns
                               if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64]]

        technical_data = df[self.technical_cols].fillna(0).values

        if fit_scaler:
            self.scaler = StandardScaler()
            self.technical_features = self.scaler.fit_transform(technical_data)
        elif scaler is not None:
            self.scaler = scaler
            self.technical_features = self.scaler.transform(technical_data)
        else:
            self.scaler = None
            self.technical_features = technical_data

        self.technical_features = np.nan_to_num(self.technical_features, posinf=3.0, neginf=-3.0)

        print(f"Dataset: {len(self.technical_cols)} technical features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)

        time_series = torch.tensor(
            self.df.iloc[max(0, idx - CHRONOS_CONTEXT_LENGTH):idx]['close'].values,
            dtype=torch.float32
        )

        if len(time_series) < CHRONOS_CONTEXT_LENGTH:
            padding = torch.full((CHRONOS_CONTEXT_LENGTH - len(time_series),),
                               time_series[0] if len(time_series) > 0 else 0.0)
            time_series = torch.cat([padding, time_series])
        elif len(time_series) > CHRONOS_CONTEXT_LENGTH:
            time_series = time_series[-CHRONOS_CONTEXT_LENGTH:]

        technical = torch.tensor(self.technical_features[idx], dtype=torch.float32)
        target = torch.tensor([row['target']], dtype=torch.float32)

        return image, time_series, technical, target

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def fetch_and_prepare_data(tickers, start_date, end_date, scalogram_path):
    all_data = []
    failed_tickers = []

    print(f"Fetching {start_date} to {end_date}")

    for ticker in tqdm(tickers, desc="Processing"):
        extended_start = pd.to_datetime(start_date) - timedelta(days=CHRONOS_CONTEXT_LENGTH + 100)

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=extended_start.strftime('%Y-%m-%d'),
                             end=end_date, auto_adjust=True)

            if df.empty:
                print(f"{ticker}: No data")
                failed_tickers.append(ticker)
                continue

            if len(df) < LOOKBACK_WINDOW + CHRONOS_CONTEXT_LENGTH:
                print(f"{ticker}: Insufficient data ({len(df)} rows)")
                failed_tickers.append(ticker)
                continue

            df.columns = [col.lower() for col in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index = df.index.tz_localize(None)

        except Exception as e:
            print(f"{ticker}: Fetch error - {e}")
            failed_tickers.append(ticker)
            continue

        try:
            df = TechnicalIndicators.calculate_all(df)
        except Exception as e:
            print(f"{ticker}: Indicator error - {e}")
            failed_tickers.append(ticker)
            continue

        df['target'] = (df['close'].shift(-PREDICTION_HORIZON) > df['close']).astype(int)

        initial_len = len(df)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"  {ticker}: Dropped {dropped} NaN rows")

        df = df.loc[df.index >= start_date].copy()

        if len(df) < 10:
            print(f"{ticker}: Too few rows ({len(df)})")
            failed_tickers.append(ticker)
            continue

        try:
            generate_wavelet_scalograms(df, scalogram_path, ticker)
        except Exception as e:
            print(f"{ticker}: Scalogram error - {e}")
            failed_tickers.append(ticker)
            continue

        df['ticker'] = ticker
        df['image_path'] = df.index.strftime('%Y-%m-%d').map(
            lambda x: os.path.join(scalogram_path, ticker, f"{x}.png")
        )

        initial_image_count = len(df)
        exists_mask = df['image_path'].apply(os.path.exists)
        missing_images = (~exists_mask).sum()

        if missing_images > 0:
            print(f"  {ticker}: {missing_images} images missing")

        df = df[exists_mask].copy()

        if len(df) > 0:
            print(f"{ticker}: {len(df)} samples")
            all_data.append(df)
        else:
            print(f"{ticker}: No valid images")
            failed_tickers.append(ticker)

    if failed_tickers:
        print(f"\nFailed: {', '.join(failed_tickers)}")

    if len(all_data) == 0:
        print("\nNo valid data collected")
        return None

    combined = pd.concat(all_data, ignore_index=False)
    print(f"\nTotal samples: {len(combined)}")
    return combined

def generate_wavelet_scalograms(df, output_path, ticker):
    ticker_path = os.path.join(output_path, ticker)
    os.makedirs(ticker_path, exist_ok=True)

    price_series = df['close']
    dates = df.index

    generated_count = 0
    skipped_count = 0

    for i in range(LOOKBACK_WINDOW, len(price_series)):
        current_date = dates[i]
        image_path = os.path.join(ticker_path, f"{current_date.strftime('%Y-%m-%d')}.png")

        if os.path.exists(image_path):
            skipped_count += 1
            continue

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
        img.save(image_path)
        generated_count += 1

    if generated_count > 0:
        print(f"  Generated {generated_count} images")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing")

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (images, time_series, technical, targets) in enumerate(pbar):
        images = images.to(device)
        time_series = time_series.to(device)
        technical = technical.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images, time_series, technical)
        loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

        pbar.set_postfix({'loss': loss.item()})

        if batch_idx % 10 == 0:
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds) * 100

    return avg_loss, accuracy

def evaluate(model, loader, criterion, device, split_name="VAL", analyze_confidence=True):
    model.eval()
    model.chronos_extractor.encoder.eval()
    total_loss = 0.0
    all_preds, all_probs, all_targets, all_confidences = [], [], [], []

    with torch.no_grad():
        for images, time_series, technical, targets in tqdm(loader, desc=f"Eval {split_name}"):
            images = images.to(device)
            time_series = time_series.to(device)
            technical = technical.to(device)
            targets = targets.to(device)

            logits, ts_confidence = model(images, time_series, technical, return_confidence=True)
            loss = criterion(logits, targets)

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_confidences.extend(ts_confidence.cpu().numpy().flatten())

    preds = np.array(all_preds)
    probs = np.array(all_probs)
    targets = np.array(all_targets)
    confidences = np.array(all_confidences)

    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(targets, preds) * 100,
        'f1_score': f1_score(targets, preds, zero_division=0),
        'auc_roc': roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0,
        'mean_confidence': confidences.mean(),
        'std_confidence': confidences.std()
    }

    print(f"\n{split_name} Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2f}%, "
          f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc_roc']:.4f}")
    print(f"{split_name} Confidence: {metrics['mean_confidence']:.3f} +/- {metrics['std_confidence']:.3f}")

    if analyze_confidence and len(confidences) > 0:
        print(f"\n{split_name} Confidence Analysis:")

        high_conf_threshold = np.percentile(confidences, 75)
        high_conf_mask = confidences >= high_conf_threshold
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(targets[high_conf_mask], preds[high_conf_mask]) * 100
            high_conf_f1 = f1_score(targets[high_conf_mask], preds[high_conf_mask], zero_division=0)
            print(f"  High confidence: Acc={high_conf_acc:.2f}%, F1={high_conf_f1:.3f}, N={high_conf_mask.sum()}")

        med_conf_mask = (confidences >= np.percentile(confidences, 25)) & (confidences < high_conf_threshold)
        if med_conf_mask.sum() > 0:
            med_conf_acc = accuracy_score(targets[med_conf_mask], preds[med_conf_mask]) * 100
            med_conf_f1 = f1_score(targets[med_conf_mask], preds[med_conf_mask], zero_division=0)
            print(f"  Medium confidence: Acc={med_conf_acc:.2f}%, F1={med_conf_f1:.3f}, N={med_conf_mask.sum()}")

        low_conf_mask = confidences < np.percentile(confidences, 25)
        if low_conf_mask.sum() > 0:
            low_conf_acc = accuracy_score(targets[low_conf_mask], preds[low_conf_mask]) * 100
            low_conf_f1 = f1_score(targets[low_conf_mask], preds[low_conf_mask], zero_division=0)
            print(f"  Low confidence: Acc={low_conf_acc:.2f}%, F1={low_conf_f1:.3f}, N={low_conf_mask.sum()}")

        metrics['confidence_analysis'] = {
            'high_confidence_acc': high_conf_acc if high_conf_mask.sum() > 0 else 0,
            'high_confidence_f1': high_conf_f1 if high_conf_mask.sum() > 0 else 0,
            'high_confidence_threshold': float(high_conf_threshold),
            'high_confidence_count': int(high_conf_mask.sum())
        }

    return metrics

def train_hybrid_model(train_loader, val_loader, model, device):
    print("\nTraining")

    trainable_params = [
        {'params': model.technical_encoder.parameters(), 'lr': LEARNING_RATE},
        {'params': model.fusion_module.parameters(), 'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},
        {'params': [model.modality_weights], 'lr': LEARNING_RATE * 2}
    ]

    if model.chronos_extractor.use_chronos:
        print("Including LoRA parameters")
        trainable_params.append({
            'params': model.chronos_extractor.encoder.model.parameters(),
            'lr': LEARNING_RATE * 0.5
        })
        trainable_params.append({
            'params': model.chronos_extractor.encoder.embedding_projection.parameters(),
            'lr': LEARNING_RATE
        })
        trainable_params.append({
            'params': model.chronos_extractor.encoder.confidence_head.parameters(),
            'lr': LEARNING_RATE
        })
    else:
        print("Including LSTM parameters")
        trainable_params.append({
            'params': model.chronos_extractor.encoder.parameters(),
            'lr': LEARNING_RATE
        })

    optimizer = torch.optim.AdamW(trainable_params, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

        val_metrics = evaluate(model, val_loader, criterion, device, "VAL")

        scheduler.step(val_metrics['f1_score'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_auc'].append(val_metrics['auc_roc'])

        weights = model.get_modality_weights()
        print(f"Weights: Visual={weights['visual']:.3f}, TS={weights['timeseries']:.3f}, Tech={weights['technical']:.3f}")

        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics,
                'modality_weights': weights,
                'scaler': train_loader.dataset.scaler,
                'technical_cols': train_loader.dataset.technical_cols
            }, MODEL_SAVE_PATH)
            print(f"Best model saved (F1={best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop at epoch {epoch+1}")
            break

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_PATH, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history

def main():
    print("Hybrid Stock Prediction System")
    print("EfficientNet + LoRA-Chronos/LSTM + Technical Indicators")

    os.makedirs(SCALOGRAM_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    if not os.path.exists(EFFICIENTNET_MODEL_PATH):
        raise FileNotFoundError(f"EfficientNet model not found: {EFFICIENTNET_MODEL_PATH}")

    print("\nTraining Data")
    train_df = fetch_and_prepare_data(TRAIN_TICKERS, TRAIN_START, VAL_START, SCALOGRAM_PATH)
    if train_df is None:
        raise RuntimeError("Failed to prepare training data")

    print(f"\nTrain: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Positive class: {train_df['target'].mean()*100:.1f}%")

    print("\nValidation Data")
    val_df = fetch_and_prepare_data(TRAIN_TICKERS, VAL_START, TEST_START, SCALOGRAM_PATH)
    if val_df is None:
        raise RuntimeError("Failed to prepare validation data")

    print(f"\nVal: {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
    print(f"  Positive class: {val_df['target'].mean()*100:.1f}%")

    print("\nTest Data")
    test_df = fetch_and_prepare_data(TRAIN_TICKERS, TEST_START, TEST_END, SCALOGRAM_PATH)
    if test_df is None:
        raise RuntimeError("Failed to prepare test data")

    print(f"\nTest: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
    print(f"  Positive class: {test_df['target'].mean()*100:.1f}%")

    print(f"\nSummary: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_dataset = HybridMultiModalDataset(
        train_df, SCALOGRAM_PATH, get_transforms(augment=True),
        scaler=None, fit_scaler=True
    )

    val_dataset = HybridMultiModalDataset(
        val_df, SCALOGRAM_PATH, get_transforms(augment=False),
        scaler=train_dataset.scaler, fit_scaler=False
    )

    test_dataset = HybridMultiModalDataset(
        test_df, SCALOGRAM_PATH, get_transforms(augment=False),
        scaler=train_dataset.scaler, fit_scaler=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print("\nModel Initialization")
    print_memory_usage(DEVICE, "Before model")

    num_technical_features = len(train_dataset.technical_cols)
    print(f"Technical features: {num_technical_features}")

    model = HybridFusionModel(
        efficientnet_path=EFFICIENTNET_MODEL_PATH,
        chronos_model_name=CHRONOS_MODEL_NAME,
        num_technical_features=num_technical_features
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} (~{trainable_params * 4 / 1024**2:.1f}MB)")

    print_memory_usage(DEVICE, "After model")

    model, history = train_hybrid_model(train_loader, val_loader, model, DEVICE)

    print("\nFinal Test Evaluation")

    checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, "TEST")

    final_weights = model.get_modality_weights()
    print(f"\nFinal modality weights:")
    print(f"  Visual: {final_weights['visual']:.3f}")
    print(f"  TimeSeries: {final_weights['timeseries']:.3f}")
    print(f"  Technical: {final_weights['technical']:.3f}")

    encoder_type = "LoRA-Chronos" if model.chronos_extractor.use_chronos else "LSTM"
    results = {
        'config': {
            'efficientnet_model': EFFICIENTNET_MODEL_PATH,
            'timeseries_encoder': encoder_type,
            'using_lora': model.chronos_extractor.use_chronos,
            'embedding_dim': EMBEDDING_DIM,
            'num_technical_features': num_technical_features,
        },
        'test_metrics': test_metrics,
        'modality_weights': final_weights,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(RESULTS_PATH, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete")
    print(f"Model: {MODEL_SAVE_PATH}")
    print(f"Results: {RESULTS_PATH}/final_results.json")

    return model, test_metrics

if __name__ == "__main__":
    model, test_metrics = main()
