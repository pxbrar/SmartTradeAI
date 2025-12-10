#!/usr/bin/env python3
"""
SmartTrade AI - Database Update Script

This script allows you to:
1. Fetch up to 50 years of historical data for stocks
2. Add new stocks to the database
3. Update existing stocks with latest data

Usage:
    python update_database.py                   # Update all stocks with max history
    python update_database.py AAPL GOOGL        # Update specific stocks
    python update_database.py --add AMZN META   # Add new stocks
    python update_database.py --list            # List stocks in database
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database import SmartTradeDB
from indicators import calculate_all_indicators
import yfinance as yf
import pandas as pd


def fetch_stock_data(symbol: str, years: int = 50) -> pd.DataFrame:
    """Fetch up to 50 years of stock data from Yahoo Finance."""
    print(f"Fetching {symbol} (up to {years} years)...")
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            print(f" No data available for {symbol}")
            return pd.DataFrame()
        
        # Clean up columns
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'date'
        df = df.reset_index()
        df['symbol'] = symbol
        
        print(f" {symbol}: {len(df):,} rows ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
        return df
        
    except Exception as e:
        print(f" Error fetching {symbol}: {e}")
        return pd.DataFrame()


def update_database(symbols: list, years: int = 50):
    """Update database with stock data."""
    print(" SmartTrade AI - Database Update")
    
    db = SmartTradeDB()
    conn = db._get_connection()
    cursor = conn.cursor()
    total_added = 0
    
    for symbol in symbols:
        df = fetch_stock_data(symbol, years)
        
        if df.empty:
            continue
        
        try:
            # Delete existing data for this symbol first
            cursor.execute("DELETE FROM prices WHERE symbol = ?", (symbol,))
            cursor.execute("DELETE FROM indicators WHERE symbol = ?", (symbol,))
            conn.commit()
            print(f" Cleared existing data for {symbol}")
            
            # Insert stock metadata
            db.insert_stock(symbol, name=symbol)
            
            # Insert prices
            rows = db.insert_prices(df)
            total_added += rows
            print(f" Saved {rows:,} records for {symbol}")
            
        except Exception as e:
            print(f" Error saving {symbol}: {e}")
    
    print(f"\n Total: {total_added:,} records added to database")
    
    # Show summary
    print("\n Database Summary:")
    summary = db.get_price_summary()
    for _, row in summary.iterrows():
        print(f"  • {row['symbol']}: {row['record_count']:,} rows ({row['first_date']} to {row['last_date']})")
    
    db.close()


def list_stocks():
    """List all stocks in database with stats."""
    db = SmartTradeDB()
    
    print("📋 Stocks in Database")
    
    summary = db.get_price_summary()
    
    if summary.empty:
        print("Database is empty. Run: python update_database.py --add AAPL MSFT")
    else:
        for _, row in summary.iterrows():
            print(f"\n  {row['symbol']}")
            print(f"    Records: {row['record_count']:,}")
            print(f"    Date Range: {row['first_date']} to {row['last_date']}")
            print(f"    Price Range: ${row['min_price']:.2f} - ${row['max_price']:.2f}")
    
    db.close()


def main():
    parser = argparse.ArgumentParser(description='Update SmartTrade AI Database')
    parser.add_argument('symbols', nargs='*', help='Stock symbols to update')
    parser.add_argument('--add', nargs='+', help='Add new stock symbols')
    parser.add_argument('--list', action='store_true', help='List stocks in database')
    parser.add_argument('--years', type=int, default=50, help='Years of history to fetch (default: 50)')
    parser.add_argument('--all', action='store_true', help='Update all stocks in database')
    
    args = parser.parse_args()
    
    if args.list:
        list_stocks()
        return
    
    # Determine which symbols to update
    symbols = []
    
    if args.add:
        symbols = args.add
    elif args.symbols:
        symbols = args.symbols
    elif args.all:
        db = SmartTradeDB()
        symbols = db.get_all_symbols()
        db.close()
        if not symbols:
            print("No stocks in database. Use --add to add stocks first.")
            return
    else:
        # Default: update all existing stocks
        db = SmartTradeDB()
        symbols = db.get_all_symbols()
        db.close()
        if not symbols:
            print("No stocks in database. Use --add to add stocks first.")
            print("   Example: python update_database.py --add AAPL MSFT GOOGL")
            return
    
    update_database(symbols, args.years)
    
    print("\nDone! Launch dashboard with: python dashboard/app_premium.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SmartTrade AI - Database Update Script
Fetch, add, update, and list stock data.
"""

import os, sys, argparse
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from database import SmartTradeDB

# Add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Fetch historical stock data
def fetch_stock_data(symbol, years=50):
    """Download up to 50 years of price history."""
    try:
        start = datetime.now() - timedelta(days=years*365)
        df = yf.Ticker(symbol).history(start=start, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df = df.reset_index().rename(columns=str.lower)
        df["symbol"] = symbol
        return df
    except: 
        return pd.DataFrame()

# Update database with new data
def update_database(symbols, years=50):
    """Clear old data and insert fresh price records."""
    db = SmartTradeDB()
    conn = db._get_connection(); cur = conn.cursor()
    total = 0

    for symbol in symbols:
        df = fetch_stock_data(symbol, years)
        if df.empty: continue

        cur.execute("DELETE FROM prices WHERE symbol=?", (symbol,))
        cur.execute("DELETE FROM indicators WHERE symbol=?", (symbol,))
        conn.commit()

        db.insert_stock(symbol)
        total += db.insert_prices(df)

    print(f"Added {total} new records")

# List available stocks
def list_stocks():
    """Show tickers stored in the database."""
    db = SmartTradeDB()
    summary = db.get_price_summary()
    print(summary if not summary.empty else "Database empty")
    db.close()

# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="SmartTrade DB Updater")
    parser.add_argument('symbols', nargs='*')
    parser.add_argument('--add', nargs='+')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--years', type=int, default=50)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.list:
        return list_stocks()

    # Determine operation
    symbols = args.add or args.symbols
    if args.all:
        symbols = SmartTradeDB().get_all_symbols()

    if not symbols:
        return print("⚠️ No stocks to update. Use --add to add symbols.")

    update_database(symbols, args.years)

if __name__ == "__main__":
    main()
