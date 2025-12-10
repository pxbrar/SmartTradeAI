"""
SmartTrade AI - Premium Dashboard (Alternative Version)

A visually stunning, feature-rich dashboard with modern UI/UX.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass  # dotenv not installed, use system env vars

# Fix comm module issue when running outside Jupyter
import unittest.mock as mock
sys.modules['comm'] = mock.MagicMock()

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from src.data_collection import ALL_SYMBOLS  # Keep for backwards compatibility
from src.indicators import calculate_all_indicators
from src.models.regression import train_price_model, train_direction_model
from src.models.forecasting import forecast_stock
from src.sentiment import SentimentAnalyzer, RealTimeSentiment
from src.signals import SignalGenerator
from src.models.classification import train_signal_classifier

# Try importing LLM predictor
try:
    from src.models.llm_predictor import LLMTradingPredictor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ LLM Predictor not available")

# Custom CSS for premium look
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
}

.glass-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.glow-text {
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
}

.pulse-dot {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.gradient-border {
    border-image: linear-gradient(45deg, #00d4ff, #7b2cbf) 1;
}

.neon-green { color: #00ff88 !important; }
.neon-red { color: #ff4757 !important; }
.neon-blue { color: #00d4ff !important; }
.neon-purple { color: #7b2cbf !important; }
"""

# Initialize Dash app with dark theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.CYBORG,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ],
    suppress_callback_exceptions=True
)
app.title = "SmartTrade AI | Premium"

# Cache for data
data_cache = {}

# Import database module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from database import SmartTradeDB

# Initialize database connection
db = SmartTradeDB()

def get_available_stocks() -> list:
    """Get list of stocks available in the database."""
    try:
        return db.get_all_symbols()
    except:
        return []

def load_stock_data(symbol: str) -> pd.DataFrame:
    """Load stock data from SQLite database (not live fetching)."""
    if symbol not in data_cache:
        try:
            # Get prices from database
            df_prices = db.get_prices(symbol)
            
            if df_prices.empty:
                print(f"⚠️ No data for {symbol} in database. Run 'python main.py' to populate.")
                return pd.DataFrame()
            
            # Get indicators from database (if available)
            df_indicators = db.get_indicators(symbol)
            
            # If indicators in database, merge them
            if not df_indicators.empty:
                # Merge prices and indicators on symbol and date
                df = pd.merge(df_prices, df_indicators, on=['symbol', 'date'], how='left', suffixes=('', '_ind'))
            else:
                # Calculate indicators if not in database
                df = df_prices.copy()
                df = calculate_all_indicators(df)
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            data_cache[symbol] = df
            print(f"✅ Loaded {len(df)} rows for {symbol} from database")
            
        except Exception as e:
            print(f"❌ Error loading {symbol} from database: {e}")
            return pd.DataFrame()
    
    return data_cache.get(symbol, pd.DataFrame())

def create_gradient_header():
    """Create animated gradient header."""
    return html.Div([
        html.Div([
            html.Div([
                html.Span("🤖", style={'fontSize': '2.5rem', 'marginRight': '15px'}),
                html.Div([
                    html.H1("SmartTrade AI", className="mb-0", 
                            style={'fontWeight': '700', 'letterSpacing': '-1px'}),
                    html.P("Premium Trading Intelligence Platform", 
                           className="text-muted mb-0", style={'fontSize': '0.9rem'})
                ])
            ], className="d-flex align-items-center"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("●", className="pulse-dot", style={'color': '#00ff88', 'marginRight': '8px'}),
                        html.Span("LIVE", style={'fontWeight': '600', 'color': '#00ff88'})
                    ], className="d-flex align-items-center justify-content-center"),
                    html.Span(datetime.now().strftime("%B %d, %Y"), className="text-muted d-block text-center", style={'fontSize': '0.85rem'}),
                    html.Img(
                        src="/assets/Humber_Logo.png",
                        style={'height': '40px', 'marginTop': '8px', 'filter': 'brightness(0) invert(1)', 'opacity': '0.7'}
                    )
                ], style={'textAlign': 'center'})
            ])
        ], className="d-flex justify-content-between align-items-center")
    ], style={
        'background': 'linear-gradient(90deg, rgba(0,212,255,0.1) 0%, rgba(123,44,191,0.1) 100%)',
        'padding': '20px 30px',
        'borderRadius': '16px',
        'marginBottom': '25px',
        'border': '1px solid rgba(255,255,255,0.1)'
    })

def create_quick_stats_row():
    """Create animated quick stats cards."""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line", style={'fontSize': '1.5rem', 'color': '#00d4ff'}),
                ], className="mb-2"),
                html.H3(id="stat-price", children="--", className="mb-0 neon-blue"),
                html.P("Current Price", className="text-muted mb-0", style={'fontSize': '0.8rem'})
            ], className="glass-card p-3 text-center")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-percentage", style={'fontSize': '1.5rem', 'color': '#00ff88'}),
                ], className="mb-2"),
                html.H3(id="stat-change", children="--", className="mb-0"),
                html.P("24h Change", className="text-muted mb-0", style={'fontSize': '0.8rem'})
            ], className="glass-card p-3 text-center")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-fire", style={'fontSize': '1.5rem', 'color': '#ff6b35'}),
                ], className="mb-2"),
                html.H3(id="stat-volume", children="--", className="mb-0", style={'color': '#ff6b35'}),
                html.P("Volume", className="text-muted mb-0", style={'fontSize': '0.8rem'})
            ], className="glass-card p-3 text-center")
        ], width=3),
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-bolt", style={'fontSize': '1.5rem', 'color': '#7b2cbf'}),
                ], className="mb-2"),
                html.H3(id="stat-signal", children="--", className="mb-0 neon-purple"),
                html.P("AI Signal", className="text-muted mb-0", style={'fontSize': '0.8rem'})
            ], className="glass-card p-3 text-center")
        ], width=3),
    ], className="mb-4")

def create_control_panel():
    """Create premium control panel."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Select Asset", className="text-muted mb-2", style={'fontSize': '0.85rem'}),
                dcc.Dropdown(
                    id='stock-selector',
                    options=[{'label': f"{'🇨🇦 ' if '.TO' in s else '🇺🇸 '}{s}", 'value': s} for s in get_available_stocks()] or [{'label': 'No stocks in database', 'value': None, 'disabled': True}],
                    value=None,
                    placeholder="Choose a stock...",
                    className="dash-dropdown",
                    style={'backgroundColor': 'rgba(255,255,255,0.05)'}
                )
            ], width=3),
            dbc.Col([
                html.Label("Time Range", className="text-muted mb-2", style={'fontSize': '0.85rem'}),
                dbc.ButtonGroup([
                    dbc.Button("1Y", id="btn-1y", color="secondary", size="sm", outline=True),
                    dbc.Button("5Y", id="btn-5y", color="primary", size="sm"),
                    dbc.Button("10Y", id="btn-10y", color="secondary", size="sm", outline=True),
                    dbc.Button("15Y", id="btn-15y", color="secondary", size="sm", outline=True),
                    dbc.Button("25Y", id="btn-25y", color="secondary", size="sm", outline=True),
                    dbc.Button("50Y", id="btn-50y", color="secondary", size="sm", outline=True),
                ], id="time-range-buttons", className="w-100")
            ], width=4),
            dbc.Col([
                html.Label("Chart Style", className="text-muted mb-2", style={'fontSize': '0.85rem'}),
                dbc.RadioItems(
                    id="chart-style",
                    options=[
                        {"label": "Candlestick", "value": "candle"},
                        {"label": "Line", "value": "line"},
                        {"label": "Area", "value": "area"},
                    ],
                    value="candle",
                    inline=True,
                    className="text-white"
                )
            ], width=3),
            dbc.Col([
                html.Label("Indicators", className="text-muted mb-2", style={'fontSize': '0.85rem'}),
                dcc.Dropdown(
                    id='overlay-indicators',
                    options=[
                        {'label': '📈 Moving Averages', 'value': 'ma'},
                        {'label': '📊 Bollinger Bands', 'value': 'bb'},
                        {'label': '💹 Volume Profile', 'value': 'vol'},
                    ],
                    value=['ma'],
                    multi=True,
                    placeholder="Add overlays..."
                )
            ], width=3),
        ])
    ], className="glass-card p-4 mb-4")

def create_main_chart_section():
    """Create the main chart area."""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H5("Price Action", className="mb-0", style={'fontWeight': '600'}),
                    html.Div([
                        dbc.Button("🔄", id="refresh-btn", color="link", size="sm", 
                                   style={'padding': '0 8px', 'fontSize': '1rem'}),
                    ], className="d-flex align-items-center")
                ], className="d-flex justify-content-between align-items-center mb-3"),
                dcc.Graph(id='main-chart', style={'height': '450px'}, config={'displayModeBar': False})
            ], className="glass-card p-4")
        ], width=8),
        dbc.Col([
            # AI Signal Card
            html.Div([
                html.H5("🤖 AI Recommendation", className="mb-3", style={'fontWeight': '600'}),
                html.Div(id="ai-signal-card")
            ], className="glass-card p-4 mb-3"),
            
            # Sentiment Gauge
            html.Div([
                html.Div([
                    html.H5("💬 Market Sentiment", className="mb-0", style={'fontWeight': '600'}),
                    dbc.Badge(id="sentiment-source-badge", children="Loading...", color="secondary", className="ms-2")
                ], className="d-flex align-items-center mb-3"),
                dcc.Graph(id='sentiment-gauge', style={'height': '150px'}, config={'displayModeBar': False}),
                html.Div(id="sentiment-headlines", className="mt-2", style={'maxHeight': '100px', 'overflowY': 'auto', 'fontSize': '0.75rem'})
            ], className="glass-card p-4 mb-3"),
            
            # LLM Prediction Card (NEW)
            html.Div([
                html.Div([
                    html.H5("🤖 LLM Analysis", className="mb-0", style={'fontWeight': '600'}),
                    dbc.Badge("Ollama", color="info", className="ms-2")
                ], className="d-flex align-items-center mb-3"),
                html.Div(id="llm-prediction-card"),
                dbc.Button("Get LLM Prediction", id="llm-predict-btn", color="primary", 
                          size="sm", className="w-100 mt-2", outline=True)
            ], className="glass-card p-4")
        ], width=4)
    ], className="mb-4")

def create_analysis_section():
    """Create the analysis section with multiple charts."""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("📊 Technical Indicators", className="mb-3", style={'fontWeight': '600'}),
                dbc.Tabs([
                    dbc.Tab(label="RSI", tab_id="tab-rsi"),
                    dbc.Tab(label="MACD", tab_id="tab-macd"),
                    dbc.Tab(label="Volume", tab_id="tab-volume"),
                ], id="indicator-tabs", active_tab="tab-rsi"),
                dcc.Graph(id='indicator-chart', style={'height': '250px'}, config={'displayModeBar': False})
            ], className="glass-card p-4")
        ], width=6),
        dbc.Col([
            html.Div([
                html.H5("🔮 Price Forecast", className="mb-3", style={'fontWeight': '600'}),
                dcc.Graph(id='forecast-chart', style={'height': '300px'}, config={'displayModeBar': False})
            ], className="glass-card p-4")
        ], width=6),
    ], className="mb-4")

def create_comparison_section():
    """Create stock comparison section."""
    return html.Div([
        html.H5("⚖️ Stock Comparison", className="mb-3", style={'fontWeight': '600'}),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='compare-stocks',
                    options=[{'label': s, 'value': s} for s in get_available_stocks()],
                    value=[],
                    multi=True,
                    placeholder="Select stocks to compare..."
                )
            ], width=6),
        ]),
        dcc.Graph(id='comparison-chart', style={'height': '300px'}, config={'displayModeBar': False})
    ], className="glass-card p-4 mb-4")


def create_ml_analysis_section():
    """Create ML Analysis section with all 9 ML categories."""
    
    # Category definitions with their models
    categories = [
        {"id": "basic-ml", "label": "1. Basic ML", "icon": "📈", "models": "Linear Regression"},
        {"id": "classification", "label": "2. Classification", "icon": "🎯", "models": "RF, DT, KNN"},
        {"id": "clustering", "label": "3. Clustering", "icon": "🔮", "models": "K-Means, Regime"},
        {"id": "timeseries", "label": "4. Time Series", "icon": "📊", "models": "ARIMA"},
        {"id": "ensembles", "label": "5. Ensembles", "icon": "🤝", "models": "Stack, XGB, Vote"},
        {"id": "deeplearning", "label": "6. Deep Learning", "icon": "🧠", "models": "LSTM, CNN, Trans"},
        {"id": "rl", "label": "7. RL", "icon": "🎮", "models": "DQN Agent"},
        {"id": "llm", "label": "8. LLM", "icon": "💬", "models": "Gemini/Ollama"},
    ]
    
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span("🎓", style={'fontSize': '2rem', 'marginRight': '15px'}),
                html.Div([
                    html.H4("ML Analysis", className="mb-0", style={'fontWeight': '700', 'letterSpacing': '-0.5px'}),
                    html.P("Comprehensive Machine Learning Model Evaluation", className="text-muted mb-0", style={'fontSize': '0.85rem'})
                ])
            ], className="d-flex align-items-center"),
            dbc.Button("🚀 Train All Models", id="train-ml-btn", color="primary", size="sm")
        ], className="d-flex justify-content-between align-items-center mb-4"),
        
        # Category Cards Overview (always visible)
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span(cat["icon"], style={'fontSize': '1.3rem'}),
                    html.Div([
                        html.Strong(cat["label"].split(". ")[1], style={'fontSize': '0.75rem'}),
                        html.Div(id=f"cat-{cat['id']}-status", children="--", style={'fontSize': '0.85rem', 'fontWeight': '600'})
                    ], className="ms-2")
                ], className="glass-card p-2 d-flex align-items-center", style={'background': 'rgba(255,255,255,0.02)'})
            ], width=True) for cat in categories
        ], className="mb-4 g-2"),
        
        # Tabs for each category
        dbc.Tabs([
            # Tab 1: Basic ML
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("📈 Regression Models", className="mb-3"),
                            dcc.Graph(id='regression-comparison-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("📉 Residual Analysis", className="mb-3"),
                            dcc.Graph(id='residual-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                ], className="mt-3")
            ], label="1. Basic ML", tab_id="tab-basic"),
            
            # Tab 2: Classification
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("🎯 Model Accuracy Comparison", className="mb-3"),
                            dcc.Graph(id='ml-accuracy-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.H6("🔥 Confusion Matrix (RF)", className="mb-3"),
                            dcc.Graph(id='confusion-matrix-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.H6("⚡ Feature Importance", className="mb-3"),
                            dcc.Graph(id='feature-importance-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=4),
                ], className="mt-3")
            ], label="2. Classification", tab_id="tab-class"),
            
            # Tab 3: Clustering
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("🔮 K-Means Clusters", className="mb-3"),
                            dcc.Graph(id='clustering-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("📊 Elbow Method", className="mb-3"),
                            dcc.Graph(id='elbow-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                ], className="mt-3")
            ], label="3. Clustering", tab_id="tab-cluster"),
            
            # Tab 4: Time Series
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("📊 ARIMA Forecast", className="mb-3"),
                            dcc.Graph(id='arima-forecast-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.H6("📈 Forecast Metrics", className="mb-3"),
                            html.Div(id="arima-metrics-display")
                        ], className="glass-card p-3", style={'height': '290px'})
                    ], width=4),
                ], className="mt-3")
            ], label="4. Time Series", tab_id="tab-ts"),
            
            # Tab 5: Ensembles
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("🤝 Ensemble Comparison", className="mb-3"),
                            dcc.Graph(id='ensemble-comparison-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("⚖️ Voting Weights", className="mb-3"),
                            dcc.Graph(id='voting-weights-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                ], className="mt-3")
            ], label="5. Ensembles", tab_id="tab-ensemble"),
            
            # Tab 6: Deep Learning
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("🧠 Deep Learning Models", className="mb-3"),
                            dcc.Graph(id='dl-comparison-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("📉 Training Loss", className="mb-3"),
                            dcc.Graph(id='dl-loss-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                ], className="mt-3")
            ], label="6. Deep Learning", tab_id="tab-dl"),
            
            # Tab 7: Reinforcement Learning
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("🎮 DQN Trading Agent", className="mb-3"),
                            dcc.Graph(id='rl-rewards-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("💰 Portfolio Value", className="mb-3"),
                            dcc.Graph(id='rl-portfolio-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                ], className="mt-3")
            ], label="7. RL", tab_id="tab-rl"),
            
            # Tab 8: LLM
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("💬 LLM Integration", className="mb-3"),
                            html.Div(id="llm-analysis-display")
                        ], className="glass-card p-3", style={'height': '290px'})
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("🎯 Prediction Accuracy", className="mb-3"),
                            dcc.Graph(id='llm-accuracy-chart', style={'height': '250px'}, config={'displayModeBar': False})
                        ], className="glass-card p-3")
                    ], width=6),
                ], className="mt-3")
            ], label="8. LLM", tab_id="tab-llm"),
            
        ], id="ml-analysis-tabs", active_tab="tab-class"),
        
        # Summary metrics row (hidden, for storing calculated values)
        dcc.Store(id='ml-metrics-store', data={}),
        
        # Hidden divs for metric outputs
        html.Div([
            html.Div(id="lr-r2-score", style={'display': 'none'}),
            html.Div(id="logistic-accuracy", style={'display': 'none'}),
            html.Div(id="rf-accuracy", style={'display': 'none'}),
            html.Div(id="arima-mape", style={'display': 'none'}),
        ])
        
    ], className="glass-card p-4 mb-4", style={'borderTop': '3px solid #7b2cbf'})

# Layout
app.layout = dbc.Container([
    # Font Awesome for icons
    html.Link(
        rel='stylesheet',
        href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    ),
    
    # Store for period selection
    dcc.Store(id='selected-period', data=365),
    
    # Header
    create_gradient_header(),
    
    # Quick Stats
    create_quick_stats_row(),
    
    # Control Panel
    create_control_panel(),
    
    # Main Chart Section
    create_main_chart_section(),
    
    # Analysis Section
    create_analysis_section(),
    
    # Comparison Section
    create_comparison_section(),
    
    # ML Analysis Section
    create_ml_analysis_section(),
    
    # Footer
    html.Div([
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
        html.P([
            "SmartTrade AI Premium • Humber College IEIT 4015 • ",
            html.Span("Built with 💙 using Plotly Dash", className="text-muted")
        ], className="text-center text-muted", style={'fontSize': '0.85rem'})
    ], className="mt-4"),
    
    # Store for LLM prediction
    dcc.Store(id='llm-prediction-store', data=None)
    
], fluid=True, style={'padding': '30px', 'minHeight': '100vh'})


# Callbacks
@app.callback(
    [Output('selected-period', 'data'),
     Output('btn-1y', 'color'), Output('btn-1y', 'outline'),
     Output('btn-5y', 'color'), Output('btn-5y', 'outline'),
     Output('btn-10y', 'color'), Output('btn-10y', 'outline'),
     Output('btn-15y', 'color'), Output('btn-15y', 'outline'),
     Output('btn-25y', 'color'), Output('btn-25y', 'outline'),
     Output('btn-50y', 'color'), Output('btn-50y', 'outline')],
    [Input('btn-1y', 'n_clicks'),
     Input('btn-5y', 'n_clicks'),
     Input('btn-10y', 'n_clicks'),
     Input('btn-15y', 'n_clicks'),
     Input('btn-25y', 'n_clicks'),
     Input('btn-50y', 'n_clicks')],
    prevent_initial_call=False
)
def update_period(n1, n5, n10, n15, n25, n50):
    ctx = callback_context
    # Default to 5Y selected
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == '.':
        return 5*365, 'secondary', True, 'primary', False, 'secondary', True, 'secondary', True, 'secondary', True, 'secondary', True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    period_map = {
        'btn-1y': 1*365,
        'btn-5y': 5*365,
        'btn-10y': 10*365,
        'btn-15y': 15*365,
        'btn-25y': 25*365,
        'btn-50y': 50*365
    }
    period = period_map.get(button_id, 5*365)
    
    # Set button styles: selected button is primary/not-outline, others are secondary/outline
    styles = {
        'btn-1y': ('secondary', True),
        'btn-5y': ('secondary', True),
        'btn-10y': ('secondary', True),
        'btn-15y': ('secondary', True),
        'btn-25y': ('secondary', True),
        'btn-50y': ('secondary', True),
    }
    styles[button_id] = ('primary', False)
    
    return (period, 
            styles['btn-1y'][0], styles['btn-1y'][1],
            styles['btn-5y'][0], styles['btn-5y'][1],
            styles['btn-10y'][0], styles['btn-10y'][1],
            styles['btn-15y'][0], styles['btn-15y'][1],
            styles['btn-25y'][0], styles['btn-25y'][1],
            styles['btn-50y'][0], styles['btn-50y'][1])


@app.callback(
    [Output('main-chart', 'figure'),
     Output('stat-price', 'children'),
     Output('stat-change', 'children'),
     Output('stat-change', 'style'),
     Output('stat-volume', 'children'),
     Output('stat-signal', 'children'),
     Output('ai-signal-card', 'children'),
     Output('sentiment-gauge', 'figure'),
     Output('sentiment-source-badge', 'children'),
     Output('sentiment-source-badge', 'color'),
     Output('sentiment-headlines', 'children'),
     Output('indicator-chart', 'figure'),
     Output('forecast-chart', 'figure')],
    [Input('stock-selector', 'value'),
     Input('selected-period', 'data'),
     Input('chart-style', 'value'),
     Input('overlay-indicators', 'value'),
     Input('indicator-tabs', 'active_tab'),
     Input('refresh-btn', 'n_clicks')]
)
def update_dashboard(symbol, period, chart_style, overlays, active_tab, refresh_clicks):
    """Main callback to update all dashboard components."""
    
    # Default empty state
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        annotations=[dict(text="Select a stock to begin", x=0.5, y=0.5, showarrow=False,
                         font=dict(size=16, color='#888'))]
    )
    
    if not symbol:
        welcome_signal = html.Div([
            html.H2("👋", className="text-center mb-3"),
            html.P("Select a stock from the dropdown above to see AI-powered analysis.",
                   className="text-center text-muted")
        ])
        return (empty_fig, "--", "--", {}, "--", "--", welcome_signal, 
                empty_fig, "--", "secondary", [], empty_fig, empty_fig)
    
    # Load data
    df = load_stock_data(symbol)
    if df.empty:
        return (empty_fig, "--", "--", {}, "--", "--", 
                html.P("No data available"), empty_fig, "--", "secondary", [], empty_fig, empty_fig)
    
    # Filter by period
    end_date = df['date'].max()
    start_date = end_date - timedelta(days=period)
    df_filtered = df[df['date'] >= start_date].copy()
    
    # Create main chart
    fig_main = go.Figure()
    
    if chart_style == 'candle':
        fig_main.add_trace(go.Candlestick(
            x=df_filtered['date'],
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ))
    elif chart_style == 'line':
        fig_main.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['close'],
            mode='lines', name='Price',
            line=dict(color='#00d4ff', width=2)
        ))
    else:  # area
        fig_main.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['close'],
            fill='tozeroy', name='Price',
            line=dict(color='#00d4ff'),
            fillcolor='rgba(0, 212, 255, 0.2)'
        ))
    
    # Add overlays
    if overlays and 'ma' in overlays:
        if 'sma_20' in df_filtered.columns:
            fig_main.add_trace(go.Scatter(
                x=df_filtered['date'], y=df_filtered['sma_20'],
                name='SMA 20', line=dict(color='#ff6b35', width=1, dash='dot')
            ))
        if 'sma_50' in df_filtered.columns:
            fig_main.add_trace(go.Scatter(
                x=df_filtered['date'], y=df_filtered['sma_50'],
                name='SMA 50', line=dict(color='#7b2cbf', width=1, dash='dot')
            ))
    
    if overlays and 'bb' in overlays:
        if 'bb_upper' in df_filtered.columns:
            fig_main.add_trace(go.Scatter(
                x=df_filtered['date'], y=df_filtered['bb_upper'],
                name='BB Upper', line=dict(color='rgba(255,255,255,0.3)', dash='dash')
            ))
            fig_main.add_trace(go.Scatter(
                x=df_filtered['date'], y=df_filtered['bb_lower'],
                name='BB Lower', line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
                fill='tonexty', fillcolor='rgba(255,255,255,0.05)'
            ))
    
    fig_main.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_color='black',
            font_size=12
        )
    )
    
    # Stats
    latest = df_filtered.iloc[-1]
    prev = df_filtered.iloc[-2] if len(df_filtered) > 1 else latest
    change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
    
    price_str = f"${latest['close']:.2f}"
    change_str = f"{'↑' if change_pct >= 0 else '↓'} {abs(change_pct):.2f}%"
    change_style = {'color': '#00ff88' if change_pct >= 0 else '#ff4757'}
    volume_str = f"{latest['volume']/1e6:.1f}M"
    
    # AI Signal
    generator = SignalGenerator()
    signal, score = generator.generate_technical_signal(latest)
    
    signal_colors = {'BUY': '#00ff88', 'SELL': '#ff4757', 'HOLD': '#ffd93d'}
    signal_icons = {'BUY': '🚀', 'SELL': '📉', 'HOLD': '⏸️'}
    
    ai_card = html.Div([
        html.Div([
            html.Span(signal_icons.get(signal, '❓'), style={'fontSize': '2rem'}),
            html.H2(signal, className="mb-0", style={'color': signal_colors.get(signal, '#fff')})
        ], className="text-center mb-3"),
        dbc.Progress(
            value=abs(score) * 100,
            color="success" if signal == "BUY" else "danger" if signal == "SELL" else "warning",
            className="mb-2",
            style={'height': '8px'}
        ),
        html.P(f"Confidence: {abs(score)*100:.0f}%", className="text-center text-muted mb-0")
    ])
    
    # Sentiment Gauge - Now using real-time data with fallback
    rt_sentiment = RealTimeSentiment()
    sentiment_data = rt_sentiment.get_realtime_sentiment(symbol)
    
    # Determine badge color based on source
    source_colors = {
        'Yahoo Finance': 'info',
        'Finnhub': 'success', 
        'Alpha Vantage': 'primary',
        'Sample Data': 'warning'
    }
    sentiment_source = sentiment_data['source']
    sentiment_badge_color = source_colors.get(sentiment_source, 'secondary')
    
    fig_sentiment = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_data['avg_score'] * 50 + 50,  # Scale -1 to 1 -> 0 to 100
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'showticklabels': False},
            'bar': {'color': '#00d4ff'},
            'bgcolor': 'rgba(255,255,255,0.05)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': 'rgba(255,71,87,0.3)'},
                {'range': [35, 65], 'color': 'rgba(255,217,61,0.3)'},
                {'range': [65, 100], 'color': 'rgba(0,255,136,0.3)'}
            ]
        },
        number={'suffix': '', 'font': {'size': 24}}
    ))
    fig_sentiment.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=150
    )
    
    # Create headlines display (show top 3 headlines)
    sentiment_headlines_display = []
    label_colors = {'positive': '#00ff88', 'negative': '#ff4757', 'neutral': '#888'}
    for item in sentiment_data.get('headlines', [])[:3]:
        sentiment_headlines_display.append(
            html.Div([
                html.Span('● ', style={'color': label_colors.get(item['label'], '#888')}),
                html.Span(item['headline'][:60] + ('...' if len(item['headline']) > 60 else ''),
                         className="text-muted")
            ], style={'marginBottom': '4px'})
        )
    
    # Indicator Chart
    fig_indicator = go.Figure()
    if active_tab == 'tab-rsi' and 'rsi_14' in df_filtered.columns:
        fig_indicator.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['rsi_14'],
            fill='tozeroy', name='RSI',
            line=dict(color='#7b2cbf'),
            fillcolor='rgba(123,44,191,0.2)'
        ))
        fig_indicator.add_hline(y=70, line_dash="dash", line_color="#ff4757")
        fig_indicator.add_hline(y=30, line_dash="dash", line_color="#00ff88")
    elif active_tab == 'tab-macd' and 'macd' in df_filtered.columns:
        fig_indicator.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['macd'],
            name='MACD', line=dict(color='#00d4ff')
        ))
        if 'macd_signal' in df_filtered.columns:
            fig_indicator.add_trace(go.Scatter(
                x=df_filtered['date'], y=df_filtered['macd_signal'],
                name='Signal', line=dict(color='#ff6b35')
            ))
    else:
        colors = ['#00ff88' if c >= o else '#ff4757' 
                  for c, o in zip(df_filtered['close'], df_filtered['open'])]
        fig_indicator.add_trace(go.Bar(
            x=df_filtered['date'], y=df_filtered['volume'],
            marker_color=colors, name='Volume'
        ))
    
    fig_indicator.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(orientation='h', y=1.1)
    )
    
    # Forecast Chart
    try:
        forecast_result = forecast_stock(df_filtered, steps=30)
        forecast_df = forecast_result['forecast']
        last_date = df_filtered['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=df_filtered['date'].tail(60), y=df_filtered['close'].tail(60),
            name='Historical', line=dict(color='#888')
        ))
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=forecast_df['forecast'],
            name='Forecast', line=dict(color='#00d4ff', dash='dash')
        ))
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=forecast_df['upper_ci'],
            fill=None, mode='lines', line=dict(color='rgba(0,212,255,0.2)'), showlegend=False
        ))
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=forecast_df['lower_ci'],
            fill='tonexty', mode='lines', line=dict(color='rgba(0,212,255,0.2)'),
            fillcolor='rgba(0,212,255,0.1)', name='Confidence'
        ))
    except:
        fig_forecast = go.Figure()
        fig_forecast.add_annotation(text="Forecast unavailable", x=0.5, y=0.5, showarrow=False)
    
    fig_forecast.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation='h', y=1.1)
    )
    
    return (fig_main, price_str, change_str, change_style, volume_str, signal,
            ai_card, fig_sentiment, sentiment_source, sentiment_badge_color, 
            sentiment_headlines_display, fig_indicator, fig_forecast)


@app.callback(
    Output('comparison-chart', 'figure'),
    [Input('compare-stocks', 'value')]
)
def update_comparison(symbols):
    """Update stock comparison chart."""
    if not symbols or len(symbols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Select at least 2 stocks to compare", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        return fig
    
    fig = go.Figure()
    colors = ['#00d4ff', '#ff6b35', '#7b2cbf', '#00ff88', '#ffd93d']
    
    for i, sym in enumerate(symbols[:5]):
        df = load_stock_data(sym)
        if not df.empty:
            # Normalize to percentage change from start
            normalized = (df['close'] / df['close'].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=df['date'], y=normalized,
                name=sym, line=dict(color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(title='% Change', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='white',
            font_color='black',
            font_size=12
        )
    )
    
    return fig


# LLM Prediction Callback
@app.callback(
    Output('llm-prediction-card', 'children'),
    [Input('llm-predict-btn', 'n_clicks')],
    [State('stock-selector', 'value')],
    prevent_initial_call=True
)
def get_llm_prediction(n_clicks, symbol):
    """Get LLM-powered trading prediction."""
    if not symbol:
        return html.P("Select a stock first", className="text-muted text-center")
    
    # Initial loading state
    if not LLM_AVAILABLE:
        return html.Div([
            html.P("🔧 LLM not available", className="text-warning"),
            html.Small("Install: pip install google-generativeai", className="text-muted")
        ])
    
    try:
        # Load stock data
        df = load_stock_data(symbol)
        if df.empty:
            return html.P("No data available", className="text-danger")
        
        # Extract indicators
        latest = df.iloc[-1]
        indicators = {}
        for col in ['rsi_14', 'macd', 'macd_signal', 'macd_histogram', 
                    'bb_percent', 'sma_20', 'sma_50', 'momentum_10', 'volatility_20',
                    'close', 'volume', 'daily_return']:
            if col in df.columns and pd.notna(latest[col]):
                indicators[col] = float(latest[col])
        
        # Use Ollama (local, free)
        predictor = LLMTradingPredictor(provider='ollama')
        prediction = predictor.predict(symbol, indicators, df[['close', 'volume']].tail(20))
        
        # Check for errors - show actual error instead of mock data
        if 'error' in prediction:
            error_msg = prediction.get('error', 'Unknown error')
            return html.Div([
                html.H5("⚠️ LLM Service Unavailable", className="text-warning mb-2"),
                html.P(f"Error: {error_msg}", className="text-muted small"),
                html.P([
                    "To enable LLM predictions, either:",
                    html.Br(),
                    "• Install Ollama locally: ", html.Code("brew install ollama && ollama serve && ollama pull mistral"),
                    html.Br(),
                    "• Or add GEMINI_API_KEY to .env file"
                ], className="small"),
                html.Hr(),
                html.P(f"Current technical analysis for {symbol}:", className="fw-bold mt-2"),
                html.Ul([
                    html.Li(f"RSI: {indicators.get('rsi_14', 'N/A'):.1f}" if isinstance(indicators.get('rsi_14'), (int, float)) else "RSI: N/A"),
                    html.Li(f"MACD: {indicators.get('macd', 'N/A'):.4f}" if isinstance(indicators.get('macd'), (int, float)) else "MACD: N/A"),
                    html.Li(f"Price: ${indicators.get('close', 0):.2f}" if isinstance(indicators.get('close'), (int, float)) else "Price: N/A"),
                ], className="small")
            ], className="p-2")
        
        # Create display card
        signal = prediction.get('signal', 'HOLD')
        confidence = prediction.get('confidence')  # No default - show real data only
        if confidence is None:
            confidence = 0  # Will display as 0% rather than fabricated 50%
        reasoning = prediction.get('reasoning', 'No reasoning provided')
        risk = prediction.get('risk_level', 'MEDIUM')
        provider = prediction.get('provider', 'unknown')
        
        signal_colors = {'BUY': '#00ff88', 'SELL': '#ff4757', 'HOLD': '#ffd93d'}
        signal_icons = {'BUY': '🚀', 'SELL': '📉', 'HOLD': '⏸️'}
        risk_colors = {'LOW': '#00ff88', 'MEDIUM': '#ffd93d', 'HIGH': '#ff4757'}
        
        return html.Div([
            # Signal
            html.Div([
                html.Span(signal_icons.get(signal, '❓'), style={'fontSize': '1.5rem'}),
                html.H4(signal, className="mb-0 ms-2", 
                       style={'color': signal_colors.get(signal, '#fff')})
            ], className="d-flex align-items-center justify-content-center mb-2"),
            
            # Confidence bar
            dbc.Progress(
                value=confidence * 100,
                color="success" if signal == "BUY" else "danger" if signal == "SELL" else "warning",
                className="mb-2", style={'height': '6px'}
            ),
            html.Small(f"Confidence: {confidence*100:.0f}%", className="text-muted d-block text-center mb-2"),
            
            # Risk level
            html.Div([
                html.Small("Risk: ", className="text-muted"),
                html.Small(risk, style={'color': risk_colors.get(risk, '#fff')})
            ], className="text-center mb-2"),
            
            # Reasoning
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)', 'margin': '8px 0'}),
            html.Small(reasoning[:200] + "..." if len(reasoning) > 200 else reasoning, 
                      className="text-muted", style={'fontSize': '0.75rem'}),
            
            # Provider badge
            html.Div([
                dbc.Badge(f"via {provider}", color="secondary", className="mt-2")
            ], className="text-center")
        ])
        
    except Exception as e:
        return html.Div([
            html.P("⚠️ Error getting LLM prediction", className="text-warning"),
            html.Small(str(e)[:100], className="text-muted")
        ])


# ML Analysis Callback - All 9 Categories
@app.callback(
    [# Category status badges
     Output('cat-basic-ml-status', 'children'),
     Output('cat-classification-status', 'children'),
     Output('cat-clustering-status', 'children'),
     Output('cat-timeseries-status', 'children'),
     Output('cat-ensembles-status', 'children'),
     Output('cat-deeplearning-status', 'children'),
     Output('cat-rl-status', 'children'),
     Output('cat-llm-status', 'children'),
     # Tab 1: Basic ML
     Output('regression-comparison-chart', 'figure'),
     Output('residual-chart', 'figure'),
     # Tab 2: Classification
     Output('ml-accuracy-chart', 'figure'),
     Output('confusion-matrix-chart', 'figure'),
     Output('feature-importance-chart', 'figure'),
     # Tab 3: Clustering
     Output('clustering-chart', 'figure'),
     Output('elbow-chart', 'figure'),
     # Tab 4: Time Series
     Output('arima-forecast-chart', 'figure'),
     Output('arima-metrics-display', 'children'),
     # Tab 5: Ensembles
     Output('ensemble-comparison-chart', 'figure'),
     Output('voting-weights-chart', 'figure'),
     # Tab 6: Deep Learning
     Output('dl-comparison-chart', 'figure'),
     Output('dl-loss-chart', 'figure'),
     # Tab 7: RL
     Output('rl-rewards-chart', 'figure'),
     Output('rl-portfolio-chart', 'figure'),
     # Tab 8: LLM
     Output('llm-analysis-display', 'children'),
     Output('llm-accuracy-chart', 'figure'),
     # Hidden outputs
     Output('lr-r2-score', 'children'),
     Output('logistic-accuracy', 'children'),
     Output('rf-accuracy', 'children'),
     Output('arima-mape', 'children'),
    ],
    [Input('train-ml-btn', 'n_clicks')],
    [State('stock-selector', 'value')],
    prevent_initial_call=True
)
def train_all_ml_models(n_clicks, symbol):
    """Train all ML models across 9 categories for ML analysis."""
    import numpy as np
    import traceback
    
    print(f"[DEBUG] train_all_ml_models called with symbol={symbol}")
    
    def empty_figure(msg="Click 'Train All Models'"):
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(color='#888', size=11))],
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig
    
    # Default outputs - must match the 29 Output definitions
    # 8 status badges + 16 figures + 1 children (arima_metrics) + 1 children (llm_display) + 1 figure (llm) + 4 hidden strings = 29
    n_outputs = 29
    defaults = (
        ["--"] * 8 +  # 8 category status badges
        [empty_figure()] * 2 +  # Basic ML: regression, residual
        [empty_figure()] * 3 +  # Classification: accuracy, confusion, feature importance
        [empty_figure()] * 2 +  # Clustering: scatter, elbow
        [empty_figure(), None] +  # Time Series: arima chart, arima_metrics (children)
        [empty_figure()] * 2 +  # Ensembles: comparison, voting weights
        [empty_figure()] * 2 +  # Deep Learning: comparison, loss
        [empty_figure()] * 2 +  # RL: rewards, portfolio
        [None, empty_figure()] +  # LLM: display (children), accuracy chart
        ["--"] * 4  # Hidden outputs
    )
    
    if not symbol:
        return defaults
    
    df = load_stock_data(symbol)
    if df.empty or len(df) < 100:
        error_defaults = (
            ["N/A"] * 8 +
            [empty_figure("Insufficient data")] * 2 +
            [empty_figure("Insufficient data")] * 3 +
            [empty_figure("Insufficient data")] * 2 +
            [empty_figure("Insufficient data"), html.P("Insufficient data")] +
            [empty_figure("Insufficient data")] * 2 +
            [empty_figure("Insufficient data")] * 2 +
            [empty_figure("Insufficient data")] * 2 +
            [html.P("Insufficient data"), empty_figure("Insufficient data")] +
            ["N/A"] * 4
        )
        return error_defaults
    
    # ===== 1. BASIC ML =====
    try:
        from src.models.regression import StockRegressor, DirectionClassifier
        
        # Train multiple regression models
        # Train linear regression model
        lr_model = StockRegressor()
        X, y = lr_model.prepare_features(df)
        lr_m = lr_model.train(X, y)
        
        basic_status = f"R²: {lr_m.get('test_r2', 0):.2f}"
        
        # Regression metrics chart - show R², RMSE comparison  
        metrics_names = ['R² Score', 'Train R²']
        metrics_vals = [lr_m.get('test_r2', 0), lr_m.get('train_r2', 0)]
        fig_regression = go.Figure(data=[go.Bar(x=metrics_names, y=metrics_vals, marker_color=['#00d4ff', '#7b2cbf'],
                                                text=[f'{v:.3f}' for v in metrics_vals], textposition='outside')])
        fig_regression.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     yaxis=dict(title='Score', range=[0, 1]), margin=dict(l=40, r=20, t=30, b=40))
        
        # Real residuals from the model
        residuals = lr_m.get('residuals', [])[:50]  # first 50 residuals
        fig_residual = go.Figure(data=[go.Scatter(y=residuals, mode='markers', marker=dict(color='#00d4ff', size=6))])
        fig_residual.add_hline(y=0, line_dash="dash", line_color="#ff4757")
        fig_residual.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   yaxis=dict(title='Residual'), xaxis=dict(title='Sample'), margin=dict(l=40, r=20, t=20, b=40))
        print("[DEBUG] 1/8 Basic ML complete")
    except Exception as e:
        print(f"[ERROR] Basic ML failed: {e}")
        basic_status = "Error"
        fig_regression = empty_figure(str(e)[:30])
        fig_residual = empty_figure()
    
    # ===== 2. CLASSIFICATION =====
    try:
        rf_clf, rf_m = train_signal_classifier(df, model_type='random_forest')
        dt_clf, dt_m = train_signal_classifier(df, model_type='decision_tree')
        knn_clf, knn_m = train_signal_classifier(df, model_type='knn')
        
        class_status = f"{rf_m.get('test_accuracy', 0)*100:.0f}%"
        
        # Accuracy comparison
        models = ['RF', 'DT', 'KNN', 'Logistic']
        log_clf = DirectionClassifier()
        X_log, y_log = log_clf.prepare_features(df)
        log_m = log_clf.train(X_log, y_log)
        accs = [rf_m.get('test_accuracy', 0)*100, dt_m.get('test_accuracy', 0)*100, 
                knn_m.get('test_accuracy', 0)*100, log_m.get('test_accuracy', 0)*100]
        
        fig_class = go.Figure(data=[go.Bar(x=models, y=accs, marker_color=['#00ff88', '#ffd93d', '#ff6b35', '#7b2cbf'],
                                           text=[f'{v:.0f}%' for v in accs], textposition='outside')])
        fig_class.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(range=[0, 100]), margin=dict(l=40, r=20, t=30, b=40))
        
        # Confusion matrix
        cm = np.array(rf_m.get('confusion_matrix', [[0]*3]*3))
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['SELL', 'HOLD', 'BUY'], y=['SELL', 'HOLD', 'BUY'],
                                           colorscale=[[0, '#1a1a2e'], [1, '#00ff88']], text=cm, texttemplate='%{text}', showscale=False))
        fig_cm.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=20, t=20, b=40))
        
        # Feature importance
        fi = rf_m.get('feature_importance', {})
        if fi:
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:6]
            fig_fi = go.Figure(data=[go.Bar(x=[f[1] for f in sorted_fi], y=[f[0] for f in sorted_fi], orientation='h',
                                            marker=dict(color=[f[1] for f in sorted_fi], colorscale='Viridis'))])
            fig_fi.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=80, r=20, t=20, b=40))
        else:
            fig_fi = empty_figure("No feature importance")
        print("[DEBUG] 2/8 Classification complete")
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        class_status = "Error"
        fig_class, fig_cm, fig_fi = empty_figure(), empty_figure(), empty_figure()
    
    # ===== 3. CLUSTERING =====
    try:
        from src.models.clustering import StockClusterer
        clusterer = StockClusterer(n_clusters=3)
        # Simulate cluster visualization using PCA
        feature_cols = ['daily_return', 'rsi_14', 'volatility_20'] if 'rsi_14' in df.columns else ['close', 'volume']
        avail_cols = [c for c in feature_cols if c in df.columns]
        if len(avail_cols) >= 2:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            X_clust = df[avail_cols].dropna().values[-100:]
            X_scaled = StandardScaler().fit_transform(X_clust)
            from sklearn.metrics import silhouette_score
            km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
            sil_score = silhouette_score(X_scaled, km.labels_)
            cluster_status = f"k=3, Sil={sil_score:.2f}"
            
            fig_clust = go.Figure(data=[go.Scatter(x=X_scaled[:, 0], y=X_scaled[:, 1] if X_scaled.shape[1] > 1 else X_scaled[:, 0],
                                                   mode='markers', marker=dict(color=km.labels_, colorscale='Viridis', size=8))])
            fig_clust.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=20, t=20, b=40))
            
            # Elbow chart
            inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(2, 7)]
            fig_elbow = go.Figure(data=[go.Scatter(x=list(range(2, 7)), y=inertias, mode='lines+markers', 
                                                   line=dict(color='#00d4ff'), marker=dict(size=10))])
            fig_elbow.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', 
                                    xaxis=dict(title='k'), yaxis=dict(title='Inertia'), margin=dict(l=40, r=20, t=20, b=40))
        else:
            cluster_status = "N/A"
            fig_clust, fig_elbow = empty_figure(), empty_figure()
        print("[DEBUG] 3/8 Clustering complete")
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}")
        cluster_status = "Error"
        fig_clust, fig_elbow = empty_figure(), empty_figure()
    
    # ===== 4. TIME SERIES =====
    try:
        forecast_result = forecast_stock(df, steps=30)
        ts_status = f"MAPE: {forecast_result['evaluation']['mape']:.1f}%"
        
        # Forecast chart
        hist_prices = df['close'].tail(60).values
        forecast_vals = forecast_result['forecast']['forecast'].values
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(y=hist_prices, name='Historical', line=dict(color='#888')))
        fig_arima.add_trace(go.Scatter(x=list(range(60, 90)), y=forecast_vals, name='Forecast', line=dict(color='#00d4ff', dash='dash')))
        fig_arima.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=20, t=20, b=40))
        
        arima_metrics = html.Div([
            html.Div([html.Strong("MAPE: "), html.Span(f"{forecast_result['evaluation']['mape']:.2f}%", style={'color': '#00ff88'})], className="mb-2"),
            html.Div([html.Strong("RMSE: "), html.Span(f"{forecast_result['evaluation']['rmse']:.2f}")], className="mb-2"),
            html.Div([html.Strong("MAE: "), html.Span(f"{forecast_result['evaluation']['mae']:.2f}")], className="mb-2"),
            html.Div([html.Strong("Order: "), html.Span("(5, 1, 0)")], className="mb-2"),
        ])
        print("[DEBUG] 4/8 Time Series complete")
    except Exception as e:
        print(f"[ERROR] Time Series failed: {e}")
        ts_status = "Error"
        fig_arima = empty_figure()
        arima_metrics = html.P("ARIMA training failed", className="text-muted")
    
    # ===== 5. ENSEMBLES (Real Training) =====
    try:
        from src.models.advanced_ensemble import train_advanced_ensemble
        
        # Train stacking ensemble
        stack_model, stack_m = train_advanced_ensemble(df, ensemble_type='stacking')
        vote_model, vote_m = train_advanced_ensemble(df, ensemble_type='voting')
        
        ensemble_status = f"{stack_m.get('test_accuracy', 0)*100:.0f}%"
        
        # Get real base learner scores - no defaults
        base_scores = stack_m.get('base_learner_scores', {})
        rf_score = base_scores.get('rf')
        gb_score = base_scores.get('gb')
        
        ens_models = ['Stacking', 'Voting']
        ens_accs = [
            stack_m.get('test_accuracy', 0) * 100,
            vote_m.get('test_accuracy', 0) * 100
        ]
        ens_colors = ['#00d4ff', '#00ff88']
        
        # Only add base learner scores if they exist (real data)
        if rf_score is not None:
            ens_models.append('RF Base')
            ens_accs.append(rf_score * 100)
            ens_colors.append('#7b2cbf')
        if gb_score is not None:
            ens_models.append('GB Base')
            ens_accs.append(gb_score * 100)
            ens_colors.append('#ff6b35')
        
        fig_ens = go.Figure(data=[go.Bar(x=ens_models, y=ens_accs, marker_color=ens_colors,
                                         text=[f'{v:.1f}%' for v in ens_accs], textposition='outside')])
        fig_ens.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, 100]), margin=dict(l=40, r=20, t=30, b=40))
        
        # Voting weights from ensemble - only use real data
        weights = vote_m.get('weights')
        if weights:
            fig_vote = go.Figure(data=[go.Pie(labels=list(weights.keys())[:4], values=list(weights.values())[:4], hole=0.5,
                                              marker=dict(colors=['#00ff88', '#00d4ff', '#7b2cbf', '#ff6b35']))])
            fig_vote.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=20, b=20), showlegend=True)
        else:
            fig_vote = empty_figure("No voting weights available")
        print("[DEBUG] 5/8 Ensembles complete")
    except Exception as e:
        print(f"[ERROR] Ensembles failed: {e}")
        ensemble_status = "Error"
        fig_ens, fig_vote = empty_figure(f"Ensemble: {str(e)[:25]}"), empty_figure()
    
    # ===== 6. DEEP LEARNING =====
    try:
        # Deep learning requires TensorFlow
        try:
            from src.models.deep_learning import train_deep_learning_model, TF_AVAILABLE
            
            if TF_AVAILABLE:
                # Train LSTM with limited epochs - use real data only, no defaults
                lstm_result = train_deep_learning_model(df, model_type='lstm', epochs=10)
                lstm_acc_raw = lstm_result[1].get('val_accuracy')
                lstm_acc = lstm_acc_raw * 100 if lstm_acc_raw is not None else 0
                lstm_loss = lstm_result[1].get('val_loss', 0)  # 0 is valid default for loss
                
                # Train CNN
                cnn_result = train_deep_learning_model(df, model_type='cnn', epochs=10)
                cnn_acc_raw = cnn_result[1].get('val_accuracy')
                cnn_acc = cnn_acc_raw * 100 if cnn_acc_raw is not None else 0
                
                # Train Transformer 
                trans_result = train_deep_learning_model(df, model_type='transformer', epochs=10)
                trans_acc_raw = trans_result[1].get('val_accuracy')
                trans_acc = trans_acc_raw * 100 if trans_acc_raw is not None else 0
                
                dl_status = f"{lstm_acc:.0f}%"
            else:
                # TensorFlow not available
                dl_status = "TF N/A"
                lstm_acc, cnn_acc, trans_acc = 0, 0, 0
                lstm_loss = 0
        except Exception as te:
            dl_status = "TF N/A"
            lstm_acc, cnn_acc, trans_acc = 0, 0, 0
            lstm_loss = 0
        
        dl_models = ['LSTM', 'CNN 1D', 'Transformer']
        dl_accs = [lstm_acc, cnn_acc, trans_acc]
        
        fig_dl = go.Figure(data=[go.Bar(x=dl_models, y=dl_accs, marker_color=['#00d4ff', '#7b2cbf', '#00ff88'],
                                        text=[f'{v:.0f}%' if v > 0 else 'N/A' for v in dl_accs], textposition='outside')])
        fig_dl.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, 100]), margin=dict(l=40, r=20, t=30, b=40))
        
        # Loss display - show real loss if trained, otherwise empty
        if lstm_acc > 0:
            fig_loss = go.Figure(data=[go.Indicator(mode="gauge+number", value=lstm_loss,
                                                     gauge={'axis': {'range': [0, 1]}, 'bar': {'color': '#ff6b35'}})])
        else:
            fig_loss = empty_figure("TensorFlow not installed")
        fig_loss.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=30, t=30, b=30))
        print("[DEBUG] 6/8 Deep Learning complete")
    except Exception as e:
        print(f"[ERROR] Deep Learning failed: {e}")
        dl_status = "Error"
        fig_dl, fig_loss = empty_figure(f"DL: {str(e)[:25]}"), empty_figure()
    
    # ===== 7. REINFORCEMENT LEARNING (Real Training) =====
    try:
        from src.models.reinforcement_learning import TradingEnvironment, DQNAgent
        print("[DEBUG] 7/8 RL training starting...")
        
        # Use only last 200 rows to speed up training
        df_rl = df.tail(200).reset_index(drop=True)
        
        # Create environment with limited data
        env = TradingEnvironment(df_rl, initial_balance=10000)
        agent = DQNAgent(state_dim=env._get_state().shape[0], action_dim=3)
        
        # Train for only 3 episodes (quick demo) - full training takes too long
        train_results = agent.train(env, episodes=3, batch_size=16, target_update_freq=2)
        
        rl_status = f"${train_results['final_portfolio']:.0f}" if 'final_portfolio' in train_results else "Trained"
        
        # Episode rewards - use only real data, no synthetic fallbacks
        rewards = train_results.get('episode_rewards')
        portfolio_values = train_results.get('portfolio_values')
        
        if rewards and portfolio_values:
            episodes = list(range(1, len(rewards) + 1))
            fig_rl = go.Figure(data=[go.Scatter(x=episodes, y=rewards, mode='lines+markers', line=dict(color='#00ff88'), name='Reward')])
            fig_rl.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', 
                                 xaxis=dict(title='Episode'), yaxis=dict(title='Total Reward'), margin=dict(l=40, r=20, t=20, b=40))
            
            fig_port = go.Figure(data=[go.Scatter(x=episodes, y=portfolio_values, mode='lines', fill='tozeroy',
                                                  line=dict(color='#00d4ff'), fillcolor='rgba(0,212,255,0.2)')])
            fig_port.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', 
                                   xaxis=dict(title='Episode'), yaxis=dict(title='Portfolio ($)'), margin=dict(l=50, r=20, t=20, b=40))
        else:
            fig_rl = empty_figure("No reward data available")
            fig_port = empty_figure("No portfolio data available")
        print("[DEBUG] 7/8 RL complete")
    except Exception as e:
        print(f"[ERROR] RL failed: {e}")
        rl_status = "Error"
        fig_rl, fig_port = empty_figure(f"RL: {str(e)[:25]}"), empty_figure()
    
    # ===== 8. LLM (Real Check) =====
    try:
        from src.models.llm_predictor import LLMTradingPredictor, GEMINI_AVAILABLE
        
        # Check actual availability
        gemini_available = GEMINI_AVAILABLE and os.environ.get('GEMINI_API_KEY')
        ollama_available = False
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=1)
            ollama_available = resp.status_code == 200
        except:
            pass
        
        llm_status = "Available" if (gemini_available or ollama_available) else "Not configured"
        
        llm_display = html.Div([
            html.Div([
                dbc.Badge("Gemini", color="info" if gemini_available else "secondary"), 
                html.Span(" Available" if gemini_available else " No API Key", 
                         className="text-success ms-2" if gemini_available else "text-muted ms-2")
            ], className="mb-2"),
            html.Div([
                dbc.Badge("Ollama", color="success" if ollama_available else "secondary"), 
                html.Span(" Running" if ollama_available else " Not Running", 
                         className="text-info ms-2" if ollama_available else "text-muted ms-2")
            ], className="mb-2"),
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
            html.Small("LLM provides natural language reasoning for trading decisions", className="text-muted")
        ])
        
        # Show actual status in chart
        llm_vals = [1 if gemini_available else 0, 1 if ollama_available else 0]
        fig_llm = go.Figure(data=[go.Bar(x=['Gemini', 'Ollama'], y=llm_vals, 
                                         marker_color=['#00d4ff' if gemini_available else '#666', 
                                                       '#00ff88' if ollama_available else '#666'])])
        fig_llm.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', 
                              yaxis=dict(title='Status', tickvals=[0, 1], ticktext=['Off', 'On']), margin=dict(l=40, r=20, t=20, b=40))
        print("[DEBUG] 8/8 LLM complete")
    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        llm_status = "Error"
        llm_display = html.P(f"LLM error: {str(e)[:50]}", className="text-muted")
        fig_llm = empty_figure()
    
    print("[DEBUG] All ML sections complete, preparing return values")
    
    # Ensure all variables are defined (catch any missed exceptions)
    try:
        result = (
            # Category statuses (8 categories now)
            basic_status, class_status, cluster_status, ts_status, ensemble_status, dl_status, rl_status, llm_status,
            # Tab 1: Basic ML
            fig_regression, fig_residual,
            # Tab 2: Classification
            fig_class, fig_cm, fig_fi,
            # Tab 3: Clustering
            fig_clust, fig_elbow,
            # Tab 4: Time Series
            fig_arima, arima_metrics,
            # Tab 5: Ensembles
            fig_ens, fig_vote,
            # Tab 6: Deep Learning
            fig_dl, fig_loss,
            # Tab 7: RL
            fig_rl, fig_port,
            # Tab 8: LLM
            llm_display, fig_llm,
            # Hidden outputs
            f"{lr_m.get('test_r2', 0):.3f}", f"{log_m.get('test_accuracy', 0)*100:.1f}%",
            f"{rf_m.get('test_accuracy', 0)*100:.1f}%", ts_status.replace("MAPE: ", "")
        )
        print(f"[DEBUG] Return tuple has {len(result)} elements (expected 29)")
        return result
    except Exception as e:
        print(f"[ERROR] Failed to create return tuple: {e}")
        traceback.print_exc()
        # Return defaults on error
        return defaults


if __name__ == '__main__':
    print("✨ Starting SmartTrade AI Premium Dashboard...")
    print("🌐 Open http://localhost:8051 in your browser")
    print("🤖 LLM Predictions: " + ("Available (Ollama)" if LLM_AVAILABLE else "Not available"))
    app.run(debug=True, port=8051)

