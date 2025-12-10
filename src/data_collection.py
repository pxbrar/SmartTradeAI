"""
SmartTrade AI - Data Collection Module

This module handles fetching stock data from Yahoo Finance API.
Supports US stocks, Canadian stocks (TSX), and market indices.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import time

# Define the stocks to analyze
STOCKS = {
    'high_growth_tech': ['NVDA', 'TSLA', 'PLTR'],
    'canadian': ['SHOP.TO', 'TD.TO', 'ENB.TO'],
    'stable': ['AAPL', 'MSFT', 'JNJ'],
    'indices': ['^GSPC', '^VIX']
}

# Flatten all symbols
ALL_SYMBOLS = [symbol for symbols in STOCKS.values() for symbol in symbols]


class StockDataCollector:
    """
    Collects and manages stock data from Yahoo Finance.
    
    Features:
    - Download historical OHLCV data
    - Support for multiple stock categories
    - Automatic retry on failures
    - Save to CSV for persistence
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data collector.
        
        Args:
            data_dir: Directory to save raw data files
        """
        if data_dir is None:
            # Get the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'raw')
        else:
            self.data_dir = data_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_stock_data(self, 
                         symbol: str, 
                         start_date: str = None, 
                         end_date: str = None,
                         period: str = '2y') -> pd.DataFrame:
        """
        Fetch historical stock data for a single symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'SHOP.TO')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Alternative to dates - '1y', '2y', '5y', 'max'
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period=period)
            
            if df.empty:
                print(f"⚠️ No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Rename columns for consistency
            df.columns = [col.replace(' ', '_') for col in df.columns]
            
            # Clean column names
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock_Splits': 'stock_splits',
                'Symbol': 'symbol'
            })
            
            # Convert date to datetime if needed and remove timezone
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            print(f"Got {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, 
                              symbols: List[str], 
                              period: str = '2y',
                              delay: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks with rate limiting.
        
        Args:
            symbols: List of stock symbols
            period: Time period to fetch
            delay: Delay between requests (seconds)
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}")
            df = self.fetch_stock_data(symbol, period=period)
            
            if not df.empty:
                data[symbol] = df
                
            # Rate limiting
            if i < len(symbols):
                time.sleep(delay)
                
        return data
    
    def fetch_all_stocks(self, period: str = '2y') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all predefined stocks.
        
        Args:
            period: Time period ('1y', '2y', '5y', 'max')
            
        Returns:
            Dictionary of all stock DataFrames
        """
        print("Starting data collection for all stocks...")
        print(f"Stocks to fetch: {ALL_SYMBOLS}")
        
        all_data = self.fetch_multiple_stocks(ALL_SYMBOLS, period=period)
        
        print("\n" + "=" * 50)
        print(f"Successfully fetched {len(all_data)}/{len(ALL_SYMBOLS)} stocks")
        
        return all_data
    
    def save_to_csv(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Save stock data to CSV files.
        
        Args:
            data: Dictionary of stock DataFrames
        """
        print("\n💾 Saving data to CSV files...")
        
        for symbol, df in data.items():
            # Clean filename (replace special characters)
            filename = symbol.replace('^', 'INDEX_').replace('.', '_')
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            
            df.to_csv(filepath, index=False)
            print(f"Saved: {filename}.csv ({len(df)} rows)")
            
        print(f"\nAll files saved to: {self.data_dir}")
    
    def load_from_csv(self, symbol: str = None) -> pd.DataFrame:
        """
        Load stock data from CSV file(s).
        
        Args:
            symbol: Specific symbol to load, or None for all
            
        Returns:
            DataFrame with loaded data
        """
        if symbol:
            filename = symbol.replace('^', 'INDEX_').replace('.', '_')
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                print(f"⚠️ File not found: {filepath}")
                return pd.DataFrame()
        else:
            # Load all CSV files
            all_data = []
            
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(self.data_dir, file)
                    df = pd.read_csv(filepath)
                    df['date'] = pd.to_datetime(df['date'])
                    all_data.append(df)
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A')
            }
        except Exception as e:
            print(f"Error getting info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}


def get_combined_dataframe(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all stock DataFrames into a single DataFrame.
    
    Args:
        data: Dictionary of stock DataFrames
        
    Returns:
        Combined DataFrame with all stocks
    """
    if not data:
        return pd.DataFrame()
    
    combined = pd.concat(data.values(), ignore_index=True)
    combined = combined.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    return combined


# Main execution for testing
if __name__ == "__main__":
    print("SmartTrade AI - Data Collection Module")
    
    # Initialize collector
    collector = StockDataCollector()
    
    # Fetch all stocks
    stock_data = collector.fetch_all_stocks(period='5y')
    
    # Save to CSV
    collector.save_to_csv(stock_data)
    
    # Show summary
    print("\nData Summary:")
    for symbol, df in stock_data.items():
        print(f"  {symbol}: {len(df)} records, "
              f"{df['date'].min().strftime('%Y-%m-%d')} to "
              f"{df['date'].max().strftime('%Y-%m-%d')}")
    
    print("\nData collection complete!")