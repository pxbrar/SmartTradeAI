"""
Ensemble Models for Stock Trading

Combines multiple models together for better predictions.
Uses stacking, voting, and bagging techniques.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    VotingClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# try to import xgboost and lightgbm (might not be installed)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class AdvancedEnsembleClassifier:
    """
    Stacking ensemble - trains multiple models and combines their predictions.
    Uses logistic regression as the final model to combine everything.
    """
    
    def __init__(self, use_neural_meta=False):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.use_neural_meta = use_neural_meta
        
        # create all our base models
        self.base_learners = self._build_base_learners()
        self.meta_learner = self._build_meta_learner()
        self.model = self._build_stacking_classifier()
    
    def _build_base_learners(self):
        """Create the models we'll combine."""
        learners = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
        ]
        
        # add xgboost if its installed
        if XGB_AVAILABLE:
            learners.append(('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42,
                use_label_encoder=False, eval_metric='mlogloss'
            )))
        
        # add lightgbm if its installed
        if LGB_AVAILABLE:
            learners.append(('lgb', lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, random_state=42, verbose=-1
            )))
        
        return learners
    
    def _build_meta_learner(self):
        """The model that combines all the base model predictions."""
        if self.use_neural_meta:
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        return LogisticRegression(max_iter=1000, random_state=42)
    
    def _build_stacking_classifier(self):
        """Put it all together."""
        return StackingClassifier(
            estimators=self.base_learners,
            final_estimator=self.meta_learner,
            cv=5,  # 5-fold cross validation
            stack_method='predict_proba',
            n_jobs=-1
        )
    
    def prepare_features(self, df, feature_cols=None, lookahead=5, threshold=0.02):
        """Get features and create BUY/SELL/HOLD labels."""
        if feature_cols is None:
            feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram', 
                           'bb_percent', 'volatility_20', 'momentum_10', 'stoch_k']
        
        available = [c for c in feature_cols if c in df.columns]
        self.feature_names = available
        df = df.copy()
        
        # create target labels based on future returns
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['signal'] = np.where(df['future_return'] > threshold, 2,  # BUY
                       np.where(df['future_return'] < -threshold, 0,  # SELL
                                1))  # HOLD
        
        df = df.dropna(subset=['signal'] + available)
        return df[available].values, df['signal'].values.astype(int)
    
    def train(self, X, y, test_size=0.2):
        """Train the stacking ensemble."""
        # clean up data
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        
        # split data (don't shuffle - time series!)
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # scale and train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # see how we did
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'model_type': 'stacking_ensemble',
            'base_learners': [name for name, _ in self.base_learners],
            'train_accuracy': accuracy_score(y_train, self.model.predict(X_train_scaled)),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # see how each base model did on its own
        base_scores = {}
        for name, learner in self.base_learners:
            clone = learner.__class__(**learner.get_params())
            clone.fit(X_train_scaled, y_train)
            pred = clone.predict(X_test_scaled)
            base_scores[name] = accuracy_score(y_test, pred)
        metrics['base_learner_scores'] = base_scores
        
        return metrics
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("train the model first!")
        return self.model.predict(self.scaler.transform(X))


class WeightedVotingEnsemble:
    """
    Multiple models vote on the prediction.
    Better models get more voting weight.
    """
    
    def __init__(self, optimize_weights=True):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.optimize_weights = optimize_weights
        self.weights = None
        self.classifiers = self._build_classifiers()
        self.model = None
    
    def _build_classifiers(self):
        clfs = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
        ]
        if XGB_AVAILABLE:
            clfs.append(('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42,
                use_label_encoder=False, eval_metric='mlogloss'
            )))
        return clfs
    
    def _optimize_weights(self, X_val, y_val):
        """Figure out how much weight each model should get."""
        weights = []
        for name, clf in self.classifiers:
            pred = clf.predict(X_val)
            acc = accuracy_score(y_val, pred)
            # weight based on how much better than random (33%)
            weights.append(max(acc - 0.33, 0.01))
        
        # normalize to sum to 1
        total = sum(weights)
        return [w / total for w in weights]
    
    def train(self, X, y, test_size=0.2):
        """Train the voting ensemble."""
        mask = ~np.isnan(X).any(axis=1)
        X, y = X[mask], y[mask]
        
        # split into train, validation, test
        split1, split2 = int(len(X) * 0.6), int(len(X) * 0.8)
        X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
        y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # train each classifier
        for name, clf in self.classifiers:
            clf.fit(X_train_scaled, y_train)
        
        # figure out weights
        if self.optimize_weights:
            self.weights = self._optimize_weights(X_val_scaled, y_val)
        else:
            self.weights = [1.0] * len(self.classifiers)
        
        # create voting classifier
        self.model = VotingClassifier(
            estimators=self.classifiers,
            voting='soft',  # use probabilities
            weights=self.weights
        )
        
        # retrain on train+val
        X_full = np.vstack([X_train_scaled, X_val_scaled])
        y_full = np.concatenate([y_train, y_val])
        self.model.fit(X_full, y_full)
        self.is_fitted = True
        
        y_pred = self.model.predict(X_test_scaled)
        
        return {
            'model_type': 'weighted_voting_ensemble',
            'weights': dict(zip([n for n, _ in self.classifiers], self.weights)),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class BaggingEnsemble:
    """
    Bagging - train many decision trees on random subsets of data.
    Each sees different samples and features.
    """
    
    def __init__(self, n_estimators=10, max_features=0.7):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=n_estimators,
            max_samples=0.8,  # each tree sees 80% of samples
            max_features=max_features,  # and 70% of features
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X, y, test_size=0.2):
        mask = ~np.isnan(X).any(axis=1)
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
            'model_type': 'bagging_ensemble',
            'n_estimators': self.model.n_estimators,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
    
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


def train_advanced_ensemble(df, ensemble_type='stacking', lookahead=5, threshold=0.02):
    """Quick way to train an ensemble model."""
    if ensemble_type == 'stacking':
        model = AdvancedEnsembleClassifier(use_neural_meta=False)
    elif ensemble_type == 'voting':
        model = WeightedVotingEnsemble(optimize_weights=True)
    elif ensemble_type == 'bagging':
        model = BaggingEnsemble(n_estimators=10)
    else:
        raise ValueError(f"unknown ensemble type: {ensemble_type}")
    
    # prepare data
    if hasattr(model, 'prepare_features'):
        X, y = model.prepare_features(df, lookahead=lookahead, threshold=threshold)
    else:
        feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram', 
                       'bb_percent', 'volatility_20', 'momentum_10', 'stoch_k']
        available = [c for c in feature_cols if c in df.columns]
        df = df.copy()
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['signal'] = np.where(df['future_return'] > threshold, 2,
                       np.where(df['future_return'] < -threshold, 0, 1))
        df = df.dropna(subset=['signal'] + available)
        X, y = df[available].values, df['signal'].values.astype(int)
    
    metrics = model.train(X, y)
    return model, metrics


# test it with real data
if __name__ == "__main__":
    print("Testing Ensemble Models with Real Stock Data...")
    
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
    print(f"Loaded {len(df)} rows of real AAPL data\n")
    
    # Test all ensemble types
    for ens_type in ['stacking', 'voting', 'bagging']:
        print(f"Training {ens_type} ensemble...")
        model, metrics = train_advanced_ensemble(df, ens_type)
        print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['test_f1_macro']:.4f}")
        
        if 'base_learner_scores' in metrics:
            print(f"  Base learner scores: {metrics['base_learner_scores']}")
        if 'weights' in metrics:
            print(f"  Weights: {metrics['weights']}")
        print()
    
    print("Done!")

