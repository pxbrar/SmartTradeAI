"""
Classification Models for Stock Trading

This file has Random Forest and other classifiers to predict 
if we should BUY, SELL, or HOLD a stock.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class TradingSignalClassifier:
    """
    This class predicts trading signals using different ML models.
    We can use Random Forest, Decision Tree, or KNN.
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.scaler = StandardScaler()  # need to scale the features
        self.feature_names = None
        self.is_fitted = False
        
        # pick which model to use
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            # default to random forest if they pick something weird
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def prepare_features(self, df, feature_cols=None, lookahead=5, threshold=0.02):
        """
        Get the features ready for training.
        We look ahead 5 days and see if price went up or down.
        """
        if feature_cols is None:
            # these are the technical indicators we'll use
            feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram', 
                           'bb_percent', 'volatility_20', 'momentum_10', 'stoch_k']
        
        # only use columns that actually exist in the data
        available = [c for c in feature_cols if c in df.columns]
        self.feature_names = available
        df = df.copy()
        
        # calculate future return to create our target
        # if return > 2%, label as BUY (2)
        # if return < -2%, label as SELL (0)  
        # otherwise HOLD (1)
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['signal'] = np.where(df['future_return'] > threshold, 2,
                       np.where(df['future_return'] < -threshold, 0, 1))
        
        # drop any rows with missing values
        df = df.dropna(subset=['signal'] + available)
        
        return df[available].values, df['signal'].values.astype(int)
    
    def train(self, X, y, test_size=0.2):
        """Train the model and see how well it does."""
        # remove any nan values first
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        
        # split data - don't shuffle because its time series!
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # scale the features (important for some models)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # check how we did
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'model_type': self.model_type,
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train_scaled)),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # get feature importance if the model has it
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            metrics['feature_importance'] = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return metrics
    
    def fit(self, X, y):
        """Simple fit method for sklearn compatibility."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def get_feature_importance(self):
        """See which features matter most."""
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names:
                return dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(enumerate(self.model.feature_importances_))
        return {}
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_fitted:
            raise RuntimeError("gotta train the model first!")
        return self.model.predict(self.scaler.transform(X))


def train_signal_classifier(df, model_type='random_forest', lookahead=5, threshold=0.02):
    """Helper function to train a classifier quickly."""
    clf = TradingSignalClassifier(model_type=model_type)
    X, y = clf.prepare_features(df, lookahead=lookahead, threshold=threshold)
    metrics = clf.train(X, y)
    return clf, metrics


# test it out with real data
if __name__ == "__main__":
    print("Testing Classification Models with Real Stock Data...")
    
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
    
    # Test different classifiers
    for model_type in ['random_forest', 'decision_tree', 'knn']:
        clf, metrics = train_signal_classifier(df, model_type=model_type)
        print(f"{model_type}: Accuracy={metrics['test_accuracy']:.4f}, F1={metrics['test_f1_macro']:.4f}")
    
    # Show feature importance for Random Forest
    clf, metrics = train_signal_classifier(df, model_type='random_forest')
    if 'feature_importance' in metrics:
        print("\nTop Features (Random Forest):")
        sorted_features = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features[:5]:
            print(f"  {feat}: {imp:.4f}")
    
    print("Done!")

