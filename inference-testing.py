"""
Advanced Testing & Evaluation for Improved Stock Prediction Model
Includes: Trading simulation, confidence analysis, ablation studies
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Assuming improved_stock_model.py is in same directory
from improved_stock_model import (
    ImprovedHybridModel, ImprovedDataset, ImprovedTechnicalIndicators,
    get_transforms, generate_wavelet_scalograms,
    EfficientNetFeatureExtractor, DEVICE, BATCH_SIZE,
    CHRONOS_CONTEXT_LENGTH, LOOKBACK_WINDOW, SCALOGRAM_PATH,
    MODEL_SAVE_PATH, RESULTS_PATH
)

# Test configuration
TEST_SCALOGRAM_PATH = './features/test_scalograms_advanced'
TEST_RESULTS_PATH = './advanced_test_results'

UNSEEN_TICKERS = [
    "CRM", "ADBE", "ORCL", "CSCO", "QCOM",  # Tech
    "MS", "AXP", "BLK",                      # Finance  
    "NKE", "COST", "ABBV", "TMO",           # Consumer/Health
    "HON", "UPS", "LMT",                     # Industrial
]

TEST_PERIODS = {
    "Q4_2024": ("2024-10-01", "2024-12-31"),
    "Q3_2024": ("2024-07-01", "2024-09-30"),
    "Q2_2024": ("2024-04-01", "2024-06-30"),
    "Q1_2024": ("2024-01-01", "2024-03-31"),
}

# ============================================================================
# TRADING SIMULATION
# ============================================================================

def simulate_trading(predictions, actuals, prices, probabilities, 
                     initial_capital=10000, transaction_cost=0.001):
    """
    Simulate realistic trading with the model's predictions
    
    Returns:
        dict: Trading metrics including returns, Sharpe ratio, max drawdown
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long
    trades = []
    portfolio_values = [initial_capital]
    
    for i in range(len(predictions)):
        pred = predictions[i]
        actual = actuals[i]
        price = prices[i]
        prob = probabilities[i]
        
        # Entry signal: predict UP with confidence > 0.6
        if pred == 1 and prob > 0.6 and position == 0:
            # Buy
            shares = (capital * 0.95) / price  # Use 95% of capital
            cost = shares * price * (1 + transaction_cost)
            if cost <= capital:
                capital -= cost
                position = shares
                trades.append({
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'capital': capital
                })
        
        # Exit signal: predict DOWN or low confidence
        elif (pred == 0 or prob < 0.55) and position > 0:
            # Sell
            proceeds = position * price * (1 - transaction_cost)
            capital += proceeds
            trades.append({
                'type': 'SELL',
                'price': price,
                'shares': position,
                'capital': capital
            })
            position = 0
        
        # Calculate portfolio value
        if position > 0:
            portfolio_value = capital + (position * price)
        else:
            portfolio_value = capital
        
        portfolio_values.append(portfolio_value)
    
    # Close any open position
    if position > 0:
        final_proceeds = position * prices[-1] * (1 - transaction_cost)
        capital += final_proceeds
        portfolio_values[-1] = capital
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    
    # Sharpe ratio (assuming 252 trading days/year, 0% risk-free rate)
    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Win rate
    winning_trades = sum(1 for t in trades if t['type'] == 'SELL' and 
                        trades[i-1]['price'] < t['price'] for i in range(1, len(trades)))
    win_rate = winning_trades / max(1, len([t for t in trades if t['type'] == 'SELL']))
    
    # Buy-and-hold comparison
    bh_return = (prices[-1] - prices[0]) / prices[0]
    
    return {
        'final_capital': portfolio_values[-1],
        'total_return': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown * 100,
        'num_trades': len([t for t in trades if t['type'] == 'BUY']),
        'win_rate': win_rate * 100,
        'buy_hold_return': bh_return * 100,
        'excess_return': (total_return - bh_return) * 100,
        'portfolio_values': portfolio_values
    }

# ============================================================================
# DATA PREPARATION
# ============================================================================

def fetch_test_data(tickers, start_date, end_date, scalogram_path, spy_data=None):
    """Fetch and prepare test data"""
    all_dfs = []
    failed = []
    
    for ticker in tqdm(tickers, desc="Fetching"):
        extended_start = pd.to_datetime(start_date) - timedelta(days=CHRONOS_CONTEXT_LENGTH + 100)
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=extended_start.strftime('%Y-%m-%d'),
                             end=end_date, auto_adjust=True)
            
            if df.empty or len(df) < LOOKBACK_WINDOW + CHRONOS_CONTEXT_LENGTH:
                failed.append(ticker)
                continue
            
            df.columns = [col.lower() for col in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index = df.index.tz_localize(None)
            
            # Calculate technical indicators
            df = ImprovedTechnicalIndicators.calculate_all(df, spy_data)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            df = df.loc[df.index >= start_date].copy()
            
            if len(df) < 10:
                failed.append(ticker)
                continue
            
            # Generate scalograms
            generate_wavelet_scalograms(df, scalogram_path, ticker)
            
            df['ticker'] = ticker
            df['image_path'] = df.index.strftime('%Y-%m-%d').map(
                lambda x: os.path.join(scalogram_path, ticker, f"{x}.png")
            )
            
            # Keep only rows with existing images
            df = df[df['image_path'].apply(os.path.exists)].copy()
            
            if len(df) > 0:
                all_dfs.append(df)
            else:
                failed.append(ticker)
        
        except Exception as e:
            failed.append(ticker)
            continue
    
    if failed:
        print(f"⚠ Failed: {', '.join(failed)}")
    
    if len(all_dfs) == 0:
        return None, failed
    
    combined = pd.concat(all_dfs, ignore_index=False)
    return combined, failed

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_comprehensive(model, loader, device, period_name):
    """Comprehensive evaluation with all metrics"""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    all_uncertainties = []
    all_gates = []
    
    with torch.no_grad():
        for images, time_series, technical, targets in tqdm(loader, desc=f"Evaluating {period_name}"):
            images = images.to(device)
            time_series = time_series.to(device)
            technical = technical.to(device)
            
            # Get predictions with all outputs
            logits, ts_conf, uncertainty, gates = model(
                images, time_series, technical, return_all=True
            )
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            all_uncertainties.extend(uncertainty.cpu().numpy().flatten())
            all_gates.extend(gates.cpu().numpy())
    
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    targets = np.array(all_targets)
    uncertainties = np.array(all_uncertainties)
    gates = np.array(all_gates)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(targets, preds) * 100,
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1_score': f1_score(targets, preds, zero_division=0),
        'auc_roc': roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5,
        'num_samples': len(targets),
        'positive_rate': targets.mean() * 100
    }
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    
    # Confidence-based metrics
    confidence_metrics = {}
    for percentile in [90, 75, 50]:
        threshold = np.percentile(probs, percentile)
        mask = np.abs(probs - 0.5) >= (threshold - 0.5)  # High confidence predictions
        
        if mask.sum() > 10:
            confidence_metrics[f'top_{100-percentile}pct'] = {
                'count': int(mask.sum()),
                'accuracy': accuracy_score(targets[mask], preds[mask]) * 100,
                'f1': f1_score(targets[mask], preds[mask], zero_division=0),
                'precision': precision_score(targets[mask], preds[mask], zero_division=0),
                'recall': recall_score(targets[mask], preds[mask], zero_division=0)
            }
    
    # Modality importance
    avg_gates = gates.mean(axis=0)
    modality_importance = {
        'visual': float(avg_gates[0]),
        'timeseries': float(avg_gates[1]),
        'technical': float(avg_gates[2])
    }
    
    # Calibration analysis
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(probs, bins) - 1
    calibration = []
    for i in range(10):
        mask = bin_indices == i
        if mask.sum() > 0:
            actual_freq = targets[mask].mean()
            predicted_freq = probs[mask].mean()
            calibration.append({
                'bin': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                'predicted': float(predicted_freq),
                'actual': float(actual_freq),
                'count': int(mask.sum())
            })
    
    return metrics, cm, confidence_metrics, modality_importance, calibration

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comprehensive_results(results, save_path):
    """Create comprehensive visualization dashboard"""
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Performance across periods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Across Test Periods', fontsize=16, fontweight='bold')
    
    periods = list(results.keys())
    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = [results[p]['metrics'][metric] * 100 if metric != 'accuracy' 
                 else results[p]['metrics'][metric] for p in periods]
        
        bars = ax.bar(range(len(periods)), values, color=colors[idx], alpha=0.7)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=45, ha='right')
        ax.set_ylabel('Score (%)')
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=np.mean(values), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(values):.2f}%')
        ax.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_across_periods.png'), dpi=300)
    plt.close()
    
    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for period in periods:
        if 'roc_data' in results[period]:
            fpr, tpr = results[period]['roc_data']
            auc = results[period]['metrics']['auc_roc']
            ax.plot(fpr, tpr, label=f'{period} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=2)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Across Test Periods', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300)
    plt.close()
    
    # 3. Confidence analysis
    fig, axes = plt.subplots(1, len(periods), figsize=(20, 5))
    fig.suptitle('Performance by Confidence Level', fontsize=16, fontweight='bold')
    
    for idx, period in enumerate(periods):
        ax = axes[idx]
        conf_metrics = results[period]['confidence_metrics']
        
        tiers = list(conf_metrics.keys())
        accuracies = [conf_metrics[t]['accuracy'] for t in tiers]
        counts = [conf_metrics[t]['count'] for t in tiers]
        
        bars = ax.bar(range(len(tiers)), accuracies, alpha=0.7, color='steelblue')
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(period)
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample counts
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'n={count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confidence_analysis.png'), dpi=300)
    plt.close()
    
    # 4. Modality importance heatmap
    modality_data = []
    for period in periods:
        imp = results[period]['modality_importance']
        modality_data.append([imp['visual'], imp['timeseries'], imp['technical']])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(np.array(modality_data).T, cmap='YlOrRd', aspect='auto')
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Visual', 'Time Series', 'Technical'])
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_title('Modality Importance Across Test Periods', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(modality_data)):
        for j in range(3):
            text = ax.text(i, j, f'{modality_data[i][j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'modality_importance.png'), dpi=300)
    plt.close()

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    print("="*80)
    print("ADVANCED TESTING & EVALUATION")
    print("="*80)
    
    os.makedirs(TEST_SCALOGRAM_PATH, exist_ok=True)
    os.makedirs(TEST_RESULTS_PATH, exist_ok=True)
    
    # Load model
    print("\n[1/5] Loading Model")
    print("-" * 80)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_SAVE_PATH}")
    
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu', weights_only=False)
    scaler = checkpoint['scaler']
    technical_cols = checkpoint['technical_cols']
    
    print(f"✓ Model loaded")
    print(f"  Validation F1: {checkpoint['val_f1']:.4f}")
    print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Initialize model
    from improved_stock_model import EFFICIENTNET_MODEL_PATH, CHRONOS_MODEL_NAME
    
    model = ImprovedHybridModel(
        efficientnet_path=EFFICIENTNET_MODEL_PATH,
        chronos_model_name=CHRONOS_MODEL_NAME,
        num_technical_features=len(technical_cols)
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model ready on {DEVICE}")
    
    # Fetch SPY for market context
    print("\n[2/5] Fetching Market Context")
    print("-" * 80)
    try:
        spy_data = yf.Ticker("SPY").history(start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d"), auto_adjust=True)
        spy_data.columns = [col.lower() for col in spy_data.columns]
        spy_data.index = spy_data.index.tz_localize(None)
        print(f"✓ SPY data loaded: {len(spy_data)} days")
    except:
        spy_data = None
        print("⚠ SPY data unavailable")
    
    # Test on all periods
    print("\n[3/5] Testing on Multiple Periods")
    print("-" * 80)
    
    all_results = {}
    
    for period_name, (start_date, end_date) in TEST_PERIODS.items():
        print(f"\n--- Testing: {period_name} ({start_date} to {end_date}) ---")
        
        # Fetch test data
        test_df, failed = fetch_test_data(
            UNSEEN_TICKERS, start_date, end_date, TEST_SCALOGRAM_PATH, spy_data
        )
        
        if test_df is None or len(test_df) == 0:
            print(f"⚠ No data for {period_name}")
            continue
        
        print(f"✓ Collected {len(test_df)} samples from {test_df['ticker'].nunique()} tickers")
        
        # Create dataset
        test_dataset = ImprovedDataset(
            test_df, TEST_SCALOGRAM_PATH, get_transforms(augment=False),
            scaler=scaler, fit_scaler=False
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Evaluate
        metrics, cm, conf_metrics, modality_imp, calibration = evaluate_comprehensive(
            model, test_loader, DEVICE, period_name
        )
        
        # Calculate ROC curve
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for images, time_series, technical, targets in test_loader:
                images = images.to(DEVICE)
                time_series = time_series.to(DEVICE)
                technical = technical.to(DEVICE)
                logits = model(images, time_series, technical)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_targets.extend(targets.numpy().flatten())
        
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        
        # Store results
        all_results[period_name] = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'confidence_metrics': conf_metrics,
            'modality_importance': modality_imp,
            'calibration': calibration,
            'roc_data': (fpr.tolist(), tpr.tolist())
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Generate visualizations
    print("\n[4/5] Generating Visualizations")
    print("-" * 80)
    plot_comprehensive_results(all_results, TEST_RESULTS_PATH)
    print(f"✓ Visualizations saved to {TEST_RESULTS_PATH}")
    
    # Save results
    print("\n[5/5] Saving Results")
    print("-" * 80)
    
    with open(os.path.join(TEST_RESULTS_PATH, 'comprehensive_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    all_accuracies = [r['metrics']['accuracy'] for r in all_results.values()]
    all_f1s = [r['metrics']['f1_score'] for r in all_results.values()]
    all_aucs = [r['metrics']['auc_roc'] for r in all_results.values()]
    
    print(f"\nOverall Performance:")
    print(f"  Average Accuracy: {np.mean(all_accuracies):.2f}% ± {np.std(all_accuracies):.2f}%")
    print(f"  Average F1 Score: {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
    print(f"  Average AUC-ROC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    
    print(f"\nPer-Period Results:")
    for period, results in all_results.items():
        m = results['metrics']
        print(f"\n  {period}:")
        print(f"    Acc: {m['accuracy']:.2f}% | F1: {m['f1_score']:.4f} | AUC: {m['auc_roc']:.4f}")
    
    print(f"\n{'='*80}")
    print("✓ TESTING COMPLETE")
    print(f"Results saved to: {TEST_RESULTS_PATH}")
    print(f"{'='*80}")
    
    return all_results

if __name__ == "__main__":
    results = main()