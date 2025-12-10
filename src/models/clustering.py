"""
Clustering Models for Stock Analysis

K-Means clustering to group similar stocks together
and detect market regimes (bull/bear/sideways).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class StockClusterer:
    """
    Uses K-Means to group stocks into clusters.
    Stocks in the same cluster have similar characteristics.
    """
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def prepare_stock_features(self, df):
        """
        Calculate summary stats for each stock.
        We group by symbol and get averages.
        """
        if 'symbol' not in df.columns:
            raise ValueError("need a 'symbol' column!")
        
        # get average values for each stock
        features = df.groupby('symbol').agg({
            'close': ['mean', 'std'],
            'volume': 'mean',
            'daily_return': ['mean', 'std'],
            'volatility_20': 'mean',
            'rsi_14': 'mean'
        }).reset_index()
        
        # flatten the column names
        features.columns = ['symbol', 'avg_price', 'price_std', 'avg_volume',
                           'avg_return', 'return_std', 'avg_volatility', 'avg_rsi']
        
        # sharpe ratio (kinda) - higher is better
        features['sharpe'] = features['avg_return'] / (features['return_std'] + 1e-10)
        
        return features
    
    def fit(self, features_df, feature_cols=None):
        """
        Run K-Means clustering on the stocks.
        """
        if feature_cols is None:
            feature_cols = ['avg_return', 'return_std', 'avg_volatility', 'avg_rsi', 'sharpe']
        
        available = [c for c in feature_cols if c in features_df.columns]
        self.feature_names = available
        
        if len(available) < 2:
            raise ValueError("need at least 2 features to cluster!")
        
        # get the data and clean it up
        X = features_df[available].values
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        symbols = features_df.loc[mask, 'symbol'].values
        
        # scale and fit kmeans
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        labels = self.model.labels_
        
        # silhouette score tells us how good the clustering is
        # closer to 1 means better separated clusters
        silhouette = silhouette_score(X_scaled, labels)
        
        results_df = pd.DataFrame({'symbol': symbols, 'cluster': labels})
        
        return {
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette,
            'cluster_sizes': dict(zip(range(self.n_clusters), 
                                      [sum(labels == i) for i in range(self.n_clusters)])),
            'results_df': results_df
        }
    
    def predict(self, features_df):
        """Assign new stocks to clusters."""
        if not self.is_fitted:
            raise ValueError("fit the model first!")
        X = features_df[self.feature_names].values
        return self.model.predict(self.scaler.transform(X))
    
    def find_optimal_clusters(self, features_df, feature_cols=None, max_k=10):
        """
        Try different values of k and find the best one.
        Uses silhouette score to pick the winner.
        """
        if feature_cols is None:
            feature_cols = ['avg_return', 'return_std', 'avg_volatility', 'avg_rsi', 'sharpe']
        
        available = [c for c in feature_cols if c in features_df.columns]
        X = features_df[available].dropna().values
        X_scaled = StandardScaler().fit_transform(X)
        
        results = {'k_values': [], 'inertia': [], 'silhouette': []}
        
        # try k from 2 to max_k
        for k in range(2, min(max_k + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            results['k_values'].append(k)
            results['inertia'].append(kmeans.inertia_)  # for elbow method
            results['silhouette'].append(silhouette_score(X_scaled, kmeans.labels_))
        
        # pick k with highest silhouette
        best_idx = np.argmax(results['silhouette'])
        results['optimal_k'] = results['k_values'][best_idx]
        
        return results


class MarketRegimeDetector:
    """
    Figures out if the market is in bull, bear, or sideways mode.
    Uses clustering on price action features.
    """
    
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42)
        self.scaler = StandardScaler()
    
    def detect_regimes(self, df, window=20):
        """
        Look at recent returns and volatility to classify regime.
        """
        df = df.copy()
        
        # calculate features over a rolling window
        df['return_window'] = df['close'].pct_change(window)
        df['volatility_window'] = df['close'].pct_change().rolling(window).std()
        df['trend'] = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        
        features = ['return_window', 'volatility_window', 'trend']
        df_clean = df.dropna(subset=features)
        
        # cluster the data
        X = df_clean[features].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        df_clean['regime'] = self.model.labels_
        
        # figure out which cluster is bull/bear/sideways based on returns
        regime_returns = df_clean.groupby('regime')['return_window'].mean()
        sorted_regimes = regime_returns.sort_values().index.tolist()
        
        # lowest return = bear, highest = bull
        label_map = {sorted_regimes[0]: 'Bear', sorted_regimes[1]: 'Sideways', sorted_regimes[2]: 'Bull'}
        df_clean['regime_label'] = df_clean['regime'].map(label_map)
        
        return df_clean


def cluster_stocks(df, n_clusters=4):
    """Quick way to cluster a bunch of stocks."""
    clusterer = StockClusterer(n_clusters=n_clusters)
    features_df = clusterer.prepare_stock_features(df)
    results = clusterer.fit(features_df)
    return clusterer, results


# test it with data
if __name__ == "__main__":
    print("Testing Clustering with Real Stock Data...")
    
    # Import data collection and indicators
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_collection import StockDataCollector
    from src.indicators import calculate_all_indicators
    
    # Fetch data for multiple stocks
    collector = StockDataCollector()
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'AMD', 'META', 'AMZN']
    
    all_data = []
    for symbol in symbols:
        df = collector.fetch_stock_data(symbol, period='2y')
        if not df.empty:
            df = calculate_all_indicators(df)
            df['symbol'] = symbol
            all_data.append(df)
            print(f"Loaded {len(df)} rows for {symbol}")
    
    if len(all_data) < 2:
        print("Need at least 2 stocks to cluster")
        exit(1)
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows across {len(all_data)} stocks")
    
    # Run clustering
    clusterer, results = cluster_stocks(combined, n_clusters=3)
    
    print(f"\nSilhouette Score: {results['silhouette_score']:.4f}")
    print(f"Cluster Sizes: {results['cluster_sizes']}")
    print("\nStock Clusters:")
    print(results['results_df'])
    
    # Test market regime detection on one stock
    print("\n--- Market Regime Detection (AAPL) ---")
    df_aapl = all_data[0]
    detector = MarketRegimeDetector(n_regimes=3)
    df_regimes = detector.detect_regimes(df_aapl, window=20)
    regime_counts = df_regimes['regime_label'].value_counts()
    print(f"Regime Distribution:\n{regime_counts}")
    
    print("Done!")

