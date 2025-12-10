"""
SmartTrade AI - Database Module

SQLite database operations for storing and querying stock data,
technical indicators, predictions, and sentiment scores.
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import List, Optional, Tuple


class SmartTradeDB:
    """
    SQLite database handler for SmartTrade AI.
    
    Tables:
    - stocks: Stock metadata (symbol, name, sector)
    - prices: Daily OHLCV data
    - indicators: Technical indicators
    - predictions: Model predictions
    - sentiment: News sentiment scores
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(
                os.path.dirname(current_dir), 
                'database', 
                'smarttrade.db'
            )
        else:
            self.db_path = db_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self.conn = None
        self._init_database()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.conn is None:
            # check_same_thread=False allows multi-threaded access (required for Dash callbacks)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Stocks table - metadata about each stock
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                currency TEXT DEFAULT 'USD',
                exchange TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Prices table - daily OHLCV data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                dividends REAL DEFAULT 0,
                stock_splits REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date),
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)
        
        # Indicators table - technical indicators
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                ema_12 REAL,
                ema_26 REAL,
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                atr_14 REAL,
                obv REAL,
                daily_return REAL,
                volatility_20 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date),
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)
        
        # Predictions table - model predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                model_name TEXT NOT NULL,
                predicted_price REAL,
                predicted_direction TEXT,
                actual_price REAL,
                actual_direction TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, model_name),
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)
        
        # Sentiment table - news sentiment scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                headline TEXT,
                source TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)
        
        # Signals table - trading signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                signal TEXT NOT NULL,
                strength REAL,
                reasons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date),
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON indicators(symbol, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_date ON sentiment(symbol, date)")
        
        conn.commit()
        print(f"Database initialized: {self.db_path}")
        
    def insert_stock(self, symbol: str, name: str = None, sector: str = None,
                     industry: str = None, market_cap: float = None,
                     currency: str = 'USD', exchange: str = None) -> None:
        """Insert or update stock metadata."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO stocks 
            (symbol, name, sector, industry, market_cap, currency, exchange)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, name, sector, industry, market_cap, currency, exchange))
        
        conn.commit()
        
    def insert_prices(self, df: pd.DataFrame) -> int:
        """
        Insert price data from DataFrame.
        
        Args:
            df: DataFrame with columns: symbol, date, open, high, low, close, volume
            
        Returns:
            Number of rows inserted
        """
        conn = self._get_connection()
        
        # Required columns
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check columns exist
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Prepare data
        df_insert = df[required_cols].copy()
        
        # Add optional columns if present
        for col in ['dividends', 'stock_splits']:
            if col in df.columns:
                df_insert[col] = df[col]
            else:
                df_insert[col] = 0
                
        # Convert date to string format
        df_insert['date'] = pd.to_datetime(df_insert['date']).dt.strftime('%Y-%m-%d')
        
        # Insert data
        rows_before = pd.read_sql("SELECT COUNT(*) as cnt FROM prices", conn)['cnt'][0]
        
        df_insert.to_sql('prices', conn, if_exists='append', index=False,
                         method='multi', chunksize=500)
        
        rows_after = pd.read_sql("SELECT COUNT(*) as cnt FROM prices", conn)['cnt'][0]
        
        return rows_after - rows_before
    
    def insert_indicators(self, df: pd.DataFrame) -> int:
        """Insert technical indicators from DataFrame."""
        conn = self._get_connection()
        df.to_sql('indicators', conn, if_exists='append', index=False,
                  method='multi', chunksize=500)
        return len(df)
    
    def insert_predictions(self, symbol: str, date: str, model_name: str,
                          predicted_price: float = None, 
                          predicted_direction: str = None,
                          confidence: float = None) -> None:
        """Insert a model prediction."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO predictions
            (symbol, date, model_name, predicted_price, predicted_direction, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, date, model_name, predicted_price, predicted_direction, confidence))
        
        conn.commit()
        
    def insert_sentiment(self, symbol: str, date: str, headline: str,
                        source: str, sentiment_score: float,
                        sentiment_label: str) -> None:
        """Insert a sentiment record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sentiment
            (symbol, date, headline, source, sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, date, headline, source, sentiment_score, sentiment_label))
        
        conn.commit()
        
    def get_prices(self, symbol: str = None, 
                   start_date: str = None, 
                   end_date: str = None) -> pd.DataFrame:
        """
        Query price data.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter start date (optional)
            end_date: Filter end date (optional)
            
        Returns:
            DataFrame with price data
        """
        conn = self._get_connection()
        
        query = "SELECT * FROM prices WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY symbol, date"
        
        df = pd.read_sql(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_indicators(self, symbol: str = None,
                       start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """Query technical indicators."""
        conn = self._get_connection()
        
        query = "SELECT * FROM indicators WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY symbol, date"
        
        df = pd.read_sql(query, conn, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols in database."""
        conn = self._get_connection()
        df = pd.read_sql("SELECT DISTINCT symbol FROM prices ORDER BY symbol", conn)
        return df['symbol'].tolist()
    
    def get_price_summary(self) -> pd.DataFrame:
        """Get summary statistics for all stocks."""
        conn = self._get_connection()
        
        query = """
            SELECT 
                symbol,
                COUNT(*) as record_count,
                MIN(date) as first_date,
                MAX(date) as last_date,
                AVG(close) as avg_price,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(volume) as avg_volume
            FROM prices
            GROUP BY symbol
            ORDER BY symbol
        """
        
        return pd.read_sql(query, conn)
    
    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute a custom SQL query."""
        conn = self._get_connection()
        return pd.read_sql(query, conn, params=params)
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("🔒 Database connection closed")


# Convenience function (do not need to instantiate class every time)
def get_db() -> SmartTradeDB:
    """Get a database instance."""
    return SmartTradeDB()


# Main execution for testing
if __name__ == "__main__":
    print("SmartTrade AI - Database Module Test")
    
    # Initialize database
    db = SmartTradeDB()
    
    # Test insert stock
    db.insert_stock('AAPL', name='Apple Inc.', sector='Technology')
    print("Stock insert test passed")
    
    # Test query
    summary = db.get_price_summary()
    print(f"Database contains {len(summary)} symbols")
    
    db.close()
    print("\nDatabase module test complete!")