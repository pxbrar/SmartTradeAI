# SmartTrade AI

## Project Overview

SmartTrade-AI is an AI-powered stock analysis and trading signal platform. It features 8 ML model categories, real-time data fetching, sentiment analysis, and an interactive dashboard.

---

## Core Modules (`src/`)

### data_collection.py
Fetches stock data from Yahoo Finance using `yfinance`.
- `StockDataCollector`: Main class for fetching OHLCV data
- Supports multiple periods (1mo, 3mo, 1y, 5y, max)
- Saves to CSV and loads into database

### database.py
SQLite database management for persistent storage.
- `SmartTradeDB`: Database wrapper class
- Tables: `prices`, `predictions`, `sentiment`, `backtests`
- CRUD operations for all data types

### indicators.py
- Calculates technical indicators.
- **Trend**: SMA (20, 50, 200), EMA
- **Momentum**: RSI, MACD, Stochastic
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA
- `calculate_all_indicators()`: Computes all at once

### sentiment.py
Analyzes news sentiment using TextBlob + keyword matching.
- `SentimentAnalyzer`: Core sentiment engine
- `RealTimeSentiment`: Fetches real headlines from Yahoo RSS, Finnhub, Alpha Vantage
- Returns positive/negative/neutral classification with confidence scores

### signals.py
Generates trading signals from technical indicators.
- `SignalGenerator`: Combines RSI, MACD, Bollinger signals
- Returns BUY/SELL/HOLD with confidence score

### backtesting.py
Tests trading strategies on historical data.
- `Backtester`: Simulates trades with configurable parameters
- Metrics: total return, Sharpe ratio, max drawdown, win rate

### api_integration.py
Clients for external APIs.
- `AlphaVantageClient`: Stock quotes and indicators
- `NewsAPIClient`: News articles
- `FREDClient`: Economic indicators (GDP, VIX, etc.)
- `CryptoClient`: CoinGecko for crypto prices

---

## Dashboard (`dashboard/`)

### app_premium.py (Main Dashboard)
Full-featured dashboard with all 8 ML categories.

**Features:**
- Real-time price charts with candlesticks
- Technical indicator overlays (SMA, BB, RSI, MACD)
- AI trading signals with confidence
- Sentiment analysis gauge
- Price forecast visualization
- ML model training across all 8 categories
- LLM-powered predictions

**Run:** `python dashboard/app_premium.py` → Open http://localhost:8051

---

## Entry Points

### main.py
CLI for fetching data and running models.
```bash
python main.py  # Interactive menu
```

### update_database.py
Updates stock data in the SQLite database.
```bash
python update_database.py  # Refreshes all stock data
```

### llm_trading_demo.py
Standalone demo of LLM trading predictions.
```bash
python llm_trading_demo.py
```

---

## Step-by-Step Setup Guide

Follow these steps in order to run SmartTrade-AI. Each step explains **what** to do and **why** it's needed.

---

### Step 1: Install Python Dependencies

```bash
cd /Users/prabh/Downloads/SmartTrade-AI
pip install -r requirements.txt
```

**Why:** Installs all required Python packages:
- `yfinance` - Fetches stock data from Yahoo Finance
- `pandas`, `numpy` - Data manipulation and numerical operations
- `scikit-learn` - ML models (Random Forest, KNN, etc.)
- `plotly`, `dash` - Interactive dashboard visualization
- `textblob` - Sentiment analysis
- `statsmodels` - ARIMA time series forecasting
- `torch` - Deep learning (LSTM, CNN, Transformer)
- `google-generativeai` - Gemini LLM integration (optional)

---

### Step 2: Configure API Keys (`.env` file)

The `.env` file is already configured with API keys:

```
GEMINI_API_KEY=
FINNHUB_API_KEY=
ALPHA_VANTAGE_KEY=
```

**Why:** API keys enable real-time features:
- `GEMINI_API_KEY` - Powers LLM predictions with natural language reasoning
- `FINNHUB_API_KEY` - Fetches real news headlines for sentiment analysis
- `ALPHA_VANTAGE_KEY` - Provides additional stock data and indicators

Without these, the dashboard shows "Not configured" for sentiment/LLM features.

---

### Step 3: Populate the Database with Stock Data

```bash
python update_database.py
```

**Why:** Downloads historical stock data for stocks and stores it in SQLite database (`database/smarttrade.db`).

**What happens:**
1. Fetches 5 years of OHLCV data from Yahoo Finance
2. Calculates all technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. Saves to SQLite for fast dashboard loading

Without this step, the dashboard shows "No stocks in database".

---

### Step 4: Run the Dashboard

```bash
python dashboard/app_premium.py
```

**Why:** Starts the Plotly Dash web server on port 8051.

**What happens:**
1. Loads stock data from SQLite database
2. Initializes all ML models
3. Starts web server at http://localhost:8051

---

### Step 5: Open the Dashboard

Open your browser and navigate to:

```
http://localhost:8051
```

**What you'll see:**
1. Stock selector dropdown (top left)
2. Price chart with candlesticks
3. Technical indicators (RSI, MACD tabs)
4. AI trading signal with confidence score
5. Sentiment analysis gauge

---

### Step 6: Train ML Models

1. Select a stock from the dropdown (e.g., AAPL)
2. Click **"Train All Models"** button

**Why:** Trains all ML model categories on the selected stock's data:

Results appear in the tabs below the main chart.

---

### Step 7: Get LLM Predictions (Optional)

1. Select a stock
2. Click **"Get LLM Prediction"** button

**Why:** Uses Google Gemini to analyze technical indicators and provide:
- BUY/SELL/HOLD recommendation
- Confidence score
- Natural language reasoning
- Risk assessment

Requires `GEMINI_API_KEY` in `.env` file.

---

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
