"""
Technical Indicators for Stock Analysis

Calculates all the technical indicators we use for prediction:
- Moving Averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- ATR, OBV, Stochastic, etc.
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def calculate_returns(df, column='close'):
        """Calculate daily returns."""
        df = df.copy()
        df['daily_return'] = df[column].pct_change()
        df['log_return'] = np.log(df[column] / df[column].shift(1))
        return df
    
    @staticmethod
    def sma(df, periods=[20, 50, 200], column='close'):
        """Simple Moving Average - just the average of last N days."""
        df = df.copy()
        for period in periods:
            df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        return df
    
    @staticmethod
    def ema(df, periods=[12, 26], column='close'):
        """Exponential Moving Average - gives more weight to recent prices."""
        df = df.copy()
        for period in periods:
            df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def rsi(df, period=14, column='close'):
        """
        RSI - Relative Strength Index
        Goes from 0-100. Above 70 = overbought, below 30 = oversold
        """
        df = df.copy()
        delta = df[column].diff()
        
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def macd(df, fast=12, slow=26, signal=9, column='close'):
        """
        MACD - Moving Average Convergence Divergence
        When MACD crosses above signal line = bullish
        """
        df = df.copy()
        
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def bollinger_bands(df, period=20, std_dev=2.0, column='close'):
        """
        Bollinger Bands - shows if price is high or low relative to recent prices
        Price near upper band = might be overbought
        """
        df = df.copy()
        
        df['bb_middle'] = df[column].rolling(window=period).mean()
        rolling_std = df[column].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def atr(df, period=14):
        """Average True Range - measures volatility."""
        df = df.copy()
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        df['true_range'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def obv(df):
        """On-Balance Volume - volume-based trend indicator."""
        df = df.copy()
        
        price_change = df['close'].diff()
        df['obv'] = np.where(
            price_change > 0, df['volume'],
            np.where(price_change < 0, -df['volume'], 0)
        ).cumsum()
        
        return df
    
    @staticmethod
    def volatility(df, period=20):
        """Rolling volatility (std dev of returns)."""
        df = df.copy()
        
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['close'].pct_change()
        
        df[f'volatility_{period}'] = df['daily_return'].rolling(window=period).std()
        df[f'volatility_{period}_annual'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        return df
    
    @staticmethod
    def momentum(df, periods=[10, 20], column='close'):
        """Price momentum - how much price changed over N days."""
        df = df.copy()
        
        for period in periods:
            df[f'momentum_{period}'] = df[column] - df[column].shift(period)
            df[f'roc_{period}'] = (df[column] / df[column].shift(period) - 1) * 100
        
        return df
    
    @staticmethod
    def stochastic(df, k_period=14, d_period=3):
        """
        Stochastic Oscillator - where is price relative to recent highs/lows
        Above 80 = overbought, below 20 = oversold
        """
        df = df.copy()
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def price_channels(df, period=20):
        """Donchian Channels - highest high and lowest low over N days."""
        df = df.copy()
        
        df['channel_high'] = df['high'].rolling(window=period).max()
        df['channel_low'] = df['low'].rolling(window=period).min()
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
        
        return df
    
    @staticmethod
    def generate_signals(df):
        """Generate trading signals based on indicators."""
        df = df.copy()
        
        # RSI signals
        if 'rsi_14' in df.columns:
            df['rsi_signal'] = np.where(
                df['rsi_14'] < 30, 'oversold',
                np.where(df['rsi_14'] > 70, 'overbought', 'neutral')
            )
        
        # MA crossover
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_crossover'] = np.where(
                (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1)),
                'golden_cross',
                np.where(
                    (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1)),
                    'death_cross', 'none'
                )
            )
        
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_crossover'] = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
                'bullish',
                np.where(
                    (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                    'bearish', 'none'
                )
            )
        
        # BB signals
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_signal'] = np.where(
                df['close'] < df['bb_lower'], 'oversold',
                np.where(df['close'] > df['bb_upper'], 'overbought', 'neutral')
            )
        
        return df


def calculate_all_indicators(df, include_signals=True):
    """Calculate all technical indicators at once."""
    ti = TechnicalIndicators()
    
    df = ti.calculate_returns(df)
    df = ti.sma(df, periods=[20, 50, 200])
    df = ti.ema(df, periods=[12, 26])
    df = ti.rsi(df, period=14)
    df = ti.macd(df)
    df = ti.bollinger_bands(df)
    df = ti.atr(df)
    df = ti.obv(df)
    df = ti.volatility(df)
    df = ti.momentum(df)
    df = ti.stochastic(df)
    
    if include_signals:
        df = ti.generate_signals(df)
    
    return df


# test it
if __name__ == "__main__":
    print("Testing Technical Indicators with Real Stock Data...")
    
    # Import data collection
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from src.data_collection import StockDataCollector
    
    # Fetch AAPL data
    collector = StockDataCollector()
    df = collector.fetch_stock_data('AAPL', period='1y')
    
    if df.empty:
        print("Failed to fetch data, check internet connection")
        exit(1)
    
    print(f"Loaded {len(df)} rows of real AAPL data")
    
    # Calculate all indicators
    result = calculate_all_indicators(df)
    
    # Count new columns added
    original_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    new_cols = [c for c in result.columns if c not in original_cols]
    print(f"Added {len(new_cols)} indicator columns")
    
    # Show sample values
    latest = result.iloc[-1]
    print(f"\nLatest AAPL indicators:")
    print(f"  RSI(14): {latest.get('rsi_14', 'N/A'):.2f}")
    print(f"  MACD: {latest.get('macd', 'N/A'):.4f}")
    print(f"  SMA(20): ${latest.get('sma_20', 'N/A'):.2f}")
    print(f"  Bollinger %B: {latest.get('bb_percent', 'N/A'):.4f}")
    print("Done!")

