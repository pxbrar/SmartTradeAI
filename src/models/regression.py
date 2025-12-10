"""
Regression Models for Stock Price Prediction

Linear regression to predict prices and 
logistic regression to predict if price goes up or down.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class StockRegressor:
    """Linear regression for price prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.model = LinearRegression()
    
    def prepare_features(self, df, feature_cols=None, target_col='close'):
        """Prepare features and target."""
        if feature_cols is None:
            feature_cols = ['sma_20', 'sma_50', 'ema_12', 'rsi_14', 'macd', 
                           'bb_upper', 'bb_lower', 'atr_14', 'volatility_20']
        
        available = [c for c in feature_cols if c in df.columns]
        self.feature_names = available
        
        if len(available) == 0:
            raise ValueError("No valid feature columns found")
        
        return df[available].values, df[target_col].values
    
    def train(self, X, y, test_size=0.2):
        """Train and evaluate the model."""
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        # Time-based split
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Scale and train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate residuals for plotting
        residuals = y_test - y_pred_test
        
        metrics = {
            'model_type': 'linear',
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'r2_test': r2_score(y_test, y_pred_test),  # Alias
            'residuals': residuals.tolist(),  # For plotting
            'predictions': y_pred_test.tolist(),
            'actual': y_test.tolist()
        }
        
        if hasattr(self.model, 'coef_') and self.feature_names:
            metrics['coefficients'] = dict(zip(self.feature_names, self.model.coef_))
        
        return metrics
    
    def fit(self, X, y):
        """Sklearn-compatible fit."""
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not trained")
        return self.model.predict(self.scaler.transform(X))


class DirectionClassifier:
    """Logistic regression for up/down prediction."""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def prepare_features(self, df, feature_cols=None, lookahead=1):
        """Prepare features for direction prediction."""
        if feature_cols is None:
            feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram', 
                           'bb_percent', 'volatility_20', 'momentum_10', 'stoch_k']
        
        available = [c for c in feature_cols if c in df.columns]
        self.feature_names = available
        
        if len(available) == 0:
            raise ValueError("No valid features found")
        
        df = df.copy()
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['direction'] = (df['future_return'] > 0).astype(int)
        df = df.dropna(subset=['direction'] + available)
        
        return df[available].values, df['direction'].values
    
    def train(self, X, y, test_size=0.2):
        """Train and evaluate."""
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        y_pred = self.model.predict(X_test_scaled)
        
        return {
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train_scaled)),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not trained")
        return self.model.predict(self.scaler.transform(X))
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not trained")
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


def train_price_model(df):
    """Train a price prediction model."""
    model = StockRegressor()
    X, y = model.prepare_features(df)
    metrics = model.train(X, y)
    return model, metrics


def train_direction_model(df, lookahead=1):
    """Train a direction prediction model."""
    model = DirectionClassifier()
    X, y = model.prepare_features(df, lookahead=lookahead)
    metrics = model.train(X, y)
    return model, metrics


# test it with real data
if __name__ == "__main__":
    print("Testing Regression Models with Real Stock Data...")
    
    # Import data collection and indicators
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_collection import StockDataCollector
    from src.indicators import calculate_all_indicators
    
    # Fetch real AAPL data
    collector = StockDataCollector()
    df = collector.fetch_stock_data('AAPL', period='5y')
    
    if df.empty:
        print("Failed to fetch data, check internet connection")
        exit(1)
    
    # Calculate all technical indicators
    df = calculate_all_indicators(df)
    print(f"Loaded {len(df)} rows of real AAPL data")
    
    # Train price prediction model
    model, metrics = train_price_model(df)
    print(f"Linear Regression R²: {metrics['test_r2']:.4f}")
    print(f"Train RMSE: {metrics['train_rmse']:.2f}, Test RMSE: {metrics['test_rmse']:.2f}")
    
    # Train direction classification model
    clf, clf_metrics = train_direction_model(df)
    print(f"Direction Accuracy: {clf_metrics['test_accuracy']:.4f}")
    print(f"Direction F1: {clf_metrics['test_f1']:.4f}")
    print("Done!")

