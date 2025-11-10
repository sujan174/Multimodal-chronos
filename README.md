# Multimodal Stock Price Prediction System

A deep learning system for predicting stock price movements using multimodal data fusion. The system combines wavelet scalogram images, time series analysis, and technical indicators to predict next-day price direction.

## Overview

This project implements a hybrid neural network architecture that fuses three different modalities:
- **Visual Features**: Wavelet scalogram images processed through EfficientNet-B0
- **Time Series Features**: Price sequences analyzed with Chronos-T5 (with LoRA fine-tuning) or LSTM
- **Technical Indicators**: 70+ technical analysis features including RSI, MACD, Bollinger Bands, etc.

The models predict binary classification: whether stock price will go UP or DOWN on the next trading day.

## Project Structure

```
Multimodal-chronos/
├── hybrid-lora.py          # Main hybrid model with Chronos/LSTM + EfficientNet
├── visuals.py              # Alternative EfficientNet-only model with fine-tuning
├── requirements.txt        # Python dependencies
├── models/                 # Trained model weights
│   ├── multi_stock_best.pth
│   └── hybrid_fusion_model.pth
├── features/               # Generated scalogram images
├── hybrid_results/         # Training results and metrics
└── test_results_unseen/    # Test results by time period
```

## Features

### Data Processing
- Automated stock data fetching from Yahoo Finance
- Wavelet scalogram generation using Morlet, Complex Morlet, and Gaussian wavelets
- Comprehensive technical indicator calculation (70+ features)
- Time-based train/validation/test splits

### Model Architectures

#### 1. Hybrid Fusion Model (hybrid-lora.py)
- **Visual Branch**: Pre-trained frozen EfficientNet-B0 feature extractor
- **Time Series Branch**: LoRA-adapted Chronos-T5 or bidirectional LSTM encoder
- **Technical Branch**: Multi-layer feed-forward network for indicator processing
- **Fusion Layer**: Cross-attention mechanism to combine all modalities
- **Learnable Modality Weights**: Automatically balances contribution from each branch

#### 2. EfficientNet Fine-Tuned Model (visuals.py)
- Two-phase training strategy:
  - Phase 1: Train classifier head with frozen backbone
  - Phase 2: Fine-tune entire network with discriminative learning rates
- Gradient accumulation and mixed precision training
- Advanced augmentation strategies

### Technical Indicators
- **Trend**: SMA, EMA (multiple periods), Moving Average Crossovers
- **Momentum**: RSI, ROC, Momentum, Stochastic Oscillator
- **Volatility**: Bollinger Bands, ATR, Parkinson Volatility
- **Volume**: OBV, MFI, Volume Ratio
- **Pattern Recognition**: Higher Highs/Lower Lows, Gap Detection
- **Advanced**: MACD, ADX, Williams %R, CCI, Ichimoku indicators

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (optional, supports CPU and MPS)
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Multimodal-chronos.git
cd Multimodal-chronos
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Chronos (optional, for hybrid model):
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
pip install peft
```

## Usage

### Training the Hybrid Model

```bash
python hybrid-lora.py
```

This will:
1. Download historical stock data for 12 major tickers
2. Generate wavelet scalogram images
3. Calculate technical indicators
4. Train the hybrid fusion model
5. Save results to `./hybrid_results/`

### Training the EfficientNet Model

```bash
python visuals.py
```

This trains the fine-tuned EfficientNet model on 43 stock tickers with two-phase training.

### Configuration

Edit the following constants at the top of each file to customize:

```python
# Date ranges
TRAIN_START = "2022-06-01"
VAL_START = "2024-01-01"
TEST_START = "2024-07-01"
TEST_END = "2024-12-31"

# Stock tickers
TRAIN_TICKERS = ["AAPL", "MSFT", "GOOGL", ...]

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20
```

### Inference

```python
import torch
from PIL import Image
from hybrid_lora import HybridFusionModel, get_transforms

# Load model
model = HybridFusionModel(
    efficientnet_path='./models/multi_stock_best.pth',
    chronos_model_name='amazon/chronos-t5-tiny',
    num_technical_features=70
)
checkpoint = torch.load('./models/hybrid_fusion_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input data
image = Image.open('path/to/scalogram.png').convert('RGB')
transform = get_transforms(augment=False)
image_tensor = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    logits = model(image_tensor, time_series_data, technical_features)
    prob = torch.sigmoid(logits).item()
    prediction = "UP" if prob > 0.5 else "DOWN"
    print(f"Prediction: {prediction} (confidence: {prob:.2%})")
```

## Model Performance

### Hybrid Model Results
- Dataset: 12 major stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, JPM, JNJ, V, WMT, PG, XOM)
- Training Period: June 2022 - December 2024
- Test F1 Score: ~0.55-0.65 (varies by stock and market conditions)

### Key Metrics
- Accuracy: Percentage of correct predictions
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Area under ROC curve
- Per-ticker performance analysis available

## Technical Details

### Wavelet Transform
- Uses Continuous Wavelet Transform (CWT)
- Three wavelets combined as RGB channels:
  - Morlet (morl): Time-frequency localization
  - Complex Morlet (cmor): Phase information
  - Gaussian (gaus1): Smooth features
- 30-day lookback window
- 31 wavelet scales for multi-resolution analysis

### Cross-Attention Fusion
- Visual features attend to time series features
- Time series features attend to visual features
- Technical features self-attention
- Feed-forward networks with residual connections
- Layer normalization and dropout for regularization

### Training Optimizations
- AdamW optimizer with weight decay
- OneCycleLR or ReduceLROnPlateau scheduling
- Gradient clipping (norm=1.0)
- Early stopping with patience
- Mixed precision training (CUDA only)
- Memory-efficient batch processing

## File Descriptions

### hybrid-lora.py
Main implementation featuring:
- Multimodal fusion architecture
- LoRA-adapted Chronos-T5 for time series encoding
- LSTM fallback when Chronos unavailable
- Cross-attention fusion mechanism
- Comprehensive evaluation metrics

### visuals.py
EfficientNet-focused implementation:
- Two-phase fine-tuning strategy
- Progressive unfreezing
- Discriminative learning rates
- Data augmentation pipeline
- Production-ready training loop

## Limitations

- Requires significant computational resources for training
- Performance varies by market conditions and volatility
- Not suitable for high-frequency trading
- Historical performance doesn't guarantee future results
- Should not be used as sole basis for investment decisions

## Future Improvements

- Add sentiment analysis from news/social media
- Implement attention visualization
- Support for additional asset classes (crypto, forex)
- Real-time prediction API
- Ensemble methods combining multiple models
- Advanced risk management features

## Research Background

This project combines several modern deep learning techniques:
- Transfer learning with pre-trained vision models
- Parameter-efficient fine-tuning (LoRA)
- Multimodal fusion with attention mechanisms
- Time series forecasting with transformers
- Technical analysis automation

## Citation

If you use this code in your research, please cite:

```
@software{multimodal_chronos,
  title={Multimodal Stock Price Prediction System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Multimodal-chronos}
}
```

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for educational and research purposes only. Trading stocks involves risk of financial loss. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Amazon Chronos for time series foundation models
- EfficientNet by Google Research
- PyWavelets for wavelet transform implementations
- Yahoo Finance for historical stock data
- Hugging Face for model infrastructure
