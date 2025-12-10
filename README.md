# SmartTrade AI

## Quick Start

### 1. Clone and Setup

```bash
git clone [repository-url]
cd SmartTrade-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```
### 2. Install Advanced Dependencies (Optional)

```bash
# Ensemble methods
pip install xgboost lightgbm

# Deep learning
pip install tensorflow
# For Apple Silicon: pip install tensorflow-macos tensorflow-metal

# LLM features
pip install google-generativeai

# Install Ollama for free local LLM
brew install ollama
ollama serve  # Start server
ollama pull mistral  # Download model
```
### 3. Run the Dashboard

```bash
cd dashboard
python app_premium.py
# Open http://localhost:8051
```

### 4. Run the LLM Trading Demo

```bash
python llm_trading_demo.py
```

## Features

### Machine Learning Models

| Category | Models | File |
|----------|--------|------|
| **Basic ML** | Linear Regression, Logistic Regression | `regression.py` |
| **Classification** | Random Forest, Decision Tree, KNN | `classification.py` |
| **Clustering** | K-Means, Stock Grouping | `clustering.py` |
| **Time Series** | ARIMA Forecasting | `forecasting.py` |
| **Advanced Ensembles** | Stacking, XGBoost, LightGBM, Voting, Bagging | `advanced_ensemble.py` |
| **Deep Learning** | LSTM, 1D CNN, Transformer, Hybrid CNN-LSTM | `deep_learning.py` |
| **Reinforcement Learning** | DQN, Policy Gradient | `reinforcement_learning.py` |
| **LLM Predictions** | Gemini, Ollama | `llm_predictor.py` |

### Technical Indicators (30+)

- **Trend:** SMA, EMA (12, 26, 50, 200)
- **Momentum:** RSI, MACD, Stochastic Oscillator
- **Volatility:** Bollinger Bands, ATR
- **Volume:** OBV, Volume Ratio

### Free LLM Providers

| Provider | Rate Limit | API Key |
|----------|------------|---------|
| **Ollama** | Unlimited (local) | None needed |
| **Gemini** | 15 RPM, 1M tokens/month | [Get Key](https://makersuite.google.com/app/apikey) |

---

## 📁 Project Structure

```
SmartTrade-AI/
├── src/
│   ├── data_collection.py      # Yahoo Finance API
│   ├── database.py             # SQLite persistence
│   ├── indicators.py           # 30+ technical indicators
│   ├── sentiment.py            # News sentiment analysis
│   ├── signals.py              # Signal aggregation
│   ├── backtesting.py          # Strategy simulation, not enabled
│   ├── api_integration.py      # Free APIs (Alpha Vantage, FRED, CoinGecko)
│   └── models/
│       ├── regression.py           # Linear Regression
│       ├── classification.py       # Random Forest, Decision Tree
│       ├── clustering.py           # K-Means
│       ├── forecasting.py          # ARIMA
│       ├── advanced_ensemble.py    # Stacking, XGBoost, LightGBM
│       ├── deep_learning.py        # LSTM, CNN, Transformer
│       ├── reinforcement_learning.py  # DQN, Policy Gradient
│       ├── llm_predictor.py        # LLM trading predictions
│       ├── llm_assistant.py        # LLM assistant & XAI
│       ├── advanced_unsupervised.py  # Anomaly detection, PCA
│       └── automl.py               # Bayesian optimization
├── dashboard/
│   └── app_premium.py          # Premium dashboard with LLM
├── database/
│   └── smarttrade.db           # SQLite database
├── llm_trading_demo.py         # LLM demo script
└── requirements.txt            # Dependencies
```

---

## Usage Examples

### Basic Stock Analysis

```python
from src.data_collection import StockDataCollector
from src.indicators import calculate_all_indicators

# Fetch data
collector = StockDataCollector()
df = collector.fetch_stock_data('AAPL', period='5y')

# Add indicators
df = calculate_all_indicators(df)
print(df[['close', 'rsi_14', 'macd', 'bb_percent']].tail())
```

### LLM Trading Prediction

```python
from src.models.llm_predictor import LLMTradingPredictor

# Initialize with Ollama (free, local)
predictor = LLMTradingPredictor(provider='ollama')

# Get prediction
indicators = {'RSI_14': 65.5, 'MACD': 1.25, 'BB_Percent': 0.75}
prediction = predictor.predict('NVDA', indicators)

print(f"Signal: {prediction['signal']}")
print(f"Confidence: {prediction['confidence']*100:.0f}%")
print(f"Reasoning: {prediction['reasoning']}")
```

### Advanced Ensemble Training

```python
from src.models.advanced_ensemble import train_advanced_ensemble

# Train stacking ensemble
model, metrics = train_advanced_ensemble(df, ensemble_type='stacking')
print(f"Accuracy: {metrics['test_accuracy']:.2%}")
```

### Deep Learning Models

```python
from src.models.deep_learning import train_deep_learning_model

# Train LSTM
lstm_model, metrics = train_deep_learning_model(df, model_type='lstm')
print(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
```

### Reinforcement Learning

```python
from src.models.reinforcement_learning import train_rl_agent

# Train DQN agent
agent, metrics = train_rl_agent(df, agent_type='dqn', episodes=100)
print(f"Total Return: {metrics['total_return']:.2%}")
```
---

## Disclaimer

This project is for **educational purposes only**. Do not use for actual trading decisions. Past performance does not guarantee future results. Always consult a financial advisor.

---

## Troubleshooting

### Ollama not running
```bash
ollama serve  # Start the server
ollama list   # Check available models
```

### TensorFlow on Apple Silicon
```bash
pip install tensorflow-macos tensorflow-metal
```

### Missing dependencies
```bash
pip install -r requirements.txt
pip install xgboost lightgbm tensorflow
```

---

## License

MIT License - See LICENSE file for details.
# SmartTradeAI
