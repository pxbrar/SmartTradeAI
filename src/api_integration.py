"""
SmartTrade AI - Free API Integration

Free financial data APIs for enhanced analysis:
- Alpha Vantage (free tier: 25 requests/day)
- News API integration for sentiment
- Federal Reserve Economic Data (FRED)
- CoinGecko for crypto (bonus)

Course Concepts Applied:
- Real-world data integration
- API rate limiting and error handling
- Multi-source data fusion
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json
import warnings
warnings.filterwarnings('ignore')


class AlphaVantageClient:
    """
    Alpha Vantage API client for free financial data.
    
    Free tier: 25 API requests per day, 5 per minute
    
    Features:
    - Real-time and historical quotes
    - Technical indicators (pre-calculated)
    - Fundamental data
    - News sentiment
    
    Register for free API key: https://www.alphavantage.co/support/#api-key
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = "9BR06FV9HYBIEJQ2"):
        """
        Initialize Alpha Vantage client.
        
        Args:
            api_key: 9BR06FV9HYBIEJQ2
        """
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = None
    
    def _rate_limit(self):
        """Enforce rate limiting (5 requests per minute)."""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < 12:  # 60 seconds / 5 requests = 12 seconds
                time.sleep(12 - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with error handling."""
        self._rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                print(f"API Note: {data['Note']}")
            
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return {}
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote for a symbol.
        
        Returns:
            Dict with price, volume, and change data
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = self._make_request(params)
        
        if 'Global Quote' in data:
            quote = data['Global Quote']
            return {
                'symbol': quote.get('01. symbol'),
                'price': float(quote.get('05. price', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_percent': quote.get('10. change percent', '0%'),
                'volume': int(quote.get('06. volume', 0)),
                'latest_trading_day': quote.get('07. latest trading day')
            }
        return {}
    
    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """
        Get daily OHLCV data.
        
        Args:
            symbol: Stock ticker
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize
        }
        
        data = self._make_request(params)
        
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        return pd.DataFrame()
    
    def get_rsi(self, symbol: str, interval: str = 'daily', 
                time_period: int = 14) -> pd.DataFrame:
        """
        Get pre-calculated RSI from Alpha Vantage.
        
        Args:
            symbol: Stock ticker
            interval: Time interval
            time_period: RSI period
        """
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        
        if 'Technical Analysis: RSI' in data:
            df = pd.DataFrame(data['Technical Analysis: RSI']).T
            df.columns = ['rsi']
            df['rsi'] = df['rsi'].astype(float)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        return pd.DataFrame()
    
    def get_macd(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """Get pre-calculated MACD."""
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': interval,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        
        if 'Technical Analysis: MACD' in data:
            df = pd.DataFrame(data['Technical Analysis: MACD']).T
            df.columns = ['macd', 'macd_signal', 'macd_histogram']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        return pd.DataFrame()
    
    def get_news_sentiment(self, symbols: List[str] = None, 
                           topics: List[str] = None) -> List[Dict]:
        """
        Get news articles with sentiment scores.
        
        Args:
            symbols: List of tickers to filter
            topics: Topics like 'earnings', 'technology', etc.
            
        Returns:
            List of articles with sentiment data
        """
        params = {'function': 'NEWS_SENTIMENT'}
        
        if symbols:
            params['tickers'] = ','.join(symbols)
        if topics:
            params['topics'] = ','.join(topics)
        
        data = self._make_request(params)
        
        if 'feed' in data:
            articles = []
            for item in data['feed'][:10]:  # Limit to 10
                articles.append({
                    'title': item.get('title'),
                    'source': item.get('source'),
                    'time_published': item.get('time_published'),
                    'overall_sentiment': item.get('overall_sentiment_label'),
                    'sentiment_score': item.get('overall_sentiment_score'),
                    'ticker_sentiments': item.get('ticker_sentiment', [])
                })
            return articles
        return []


class NewsAPIClient:
    """
    News API client for free news data and sentiment analysis.
    
    Free tier: 100 requests/day, 1 month history
    
    Register for free API key: https://newsapi.org/register
    """
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: str = None):
        """
        Initialize News API client.
        
        Args:
            api_key: Your News API key
        """
        self.api_key = api_key
    
    def get_stock_news(self, query: str, days_back: int = 7) -> List[Dict]:
        """
        Get news articles about a stock or topic.
        
        Args:
            query: Search query (e.g., company name or ticker)
            days_back: How many days of history to fetch
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            return self._mock_news(query)
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}/everything", params=params)
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])[:10]
        except Exception as e:
            print(f"News API error: {e}")
        
        return self._mock_news(query)
    
    def _mock_news(self, query: str) -> List[Dict]:
        """Return empty list when API is not available - no fake data."""
        print(f"News API key not configured. No news data available for {query}.")
        print("   Get a free API key at: https://newsapi.org/register")
        return []


class FREDClient:
    """
    Federal Reserve Economic Data (FRED) API client.
    
    Free API with extensive economic indicators:
    - GDP, Unemployment, Inflation
    - Interest rates (Fed Funds, Treasury yields)
    - Market indicators (VIX, etc.)
    
    Register for free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Common economic indicators
    INDICATORS = {
        'GDP': 'GDP',
        'UNEMPLOYMENT': 'UNRATE',
        'INFLATION': 'CPIAUCSL',
        'FED_FUNDS': 'FEDFUNDS',
        'TREASURY_10Y': 'DGS10',
        'VIX': 'VIXCLS',
        'SP500': 'SP500'
    }
    
    def __init__(self, api_key: str = None):
        """Initialize FRED client."""
        self.api_key = api_key
    
    def get_series(self, series_id: str, 
                   start_date: str = None, 
                   end_date: str = None) -> pd.DataFrame:
        """
        Get economic time series data.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date index and values
        """
        if not self.api_key:
            return self._mock_series(series_id)
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }
        
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        try:
            response = requests.get(f"{self.BASE_URL}/series/observations", params=params)
            data = response.json()
            
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')[['value']]
                df.columns = [series_id.lower()]
                return df
        except Exception as e:
            print(f"FRED API error: {e}")
        
        return self._mock_series(series_id)
    
    def _mock_series(self, series_id: str) -> pd.DataFrame:
        """Return empty DataFrame when API is not available - no fake data."""
        print(f"⚠️ FRED API key not configured. No data available for {series_id}.")
        print("   Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return pd.DataFrame()
    
    def get_economic_context(self) -> Dict:
        """
        Get current economic context from multiple indicators.
        
        Returns:
            Dict with key economic indicators
        """
        context = {}
        
        for name, series_id in self.INDICATORS.items():
            df = self.get_series(series_id)
            if not df.empty:
                context[name] = {
                    'latest_value': float(df.iloc[-1, 0]),
                    'change_1m': float(df.iloc[-1, 0] - df.iloc[-22, 0]) if len(df) > 22 else None,
                    'series_id': series_id
                }
        
        return context


class CryptoClient:
    """
    CoinGecko API client for free cryptocurrency data.
    
    Completely free with no API key required.
    Rate limit: 10-50 calls/minute depending on endpoint.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        """Initialize CoinGecko client."""
        pass
    
    def get_price(self, coins: List[str], vs_currency: str = 'usd') -> Dict:
        """
        Get current price for cryptocurrencies.
        
        Args:
            coins: List of coin IDs (e.g., ['bitcoin', 'ethereum'])
            vs_currency: Quote currency
            
        Returns:
            Dict with prices
        """
        params = {
            'ids': ','.join(coins),
            'vs_currencies': vs_currency,
            'include_24hr_change': 'true'
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}/simple/price", params=params)
            return response.json()
        except Exception as e:
            print(f"CoinGecko error: {e}")
            return {}
    
    def get_market_data(self, coin: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical market data for a cryptocurrency.
        
        Args:
            coin: Coin ID (e.g., 'bitcoin')
            days: Number of days of history
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/coins/{coin}/ohlc",
                params={'vs_currency': 'usd', 'days': days}
            )
            data = response.json()
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            return df
        except Exception as e:
            print(f"CoinGecko error: {e}")
            return pd.DataFrame()
    
    def get_trending(self) -> List[Dict]:
        """Get trending cryptocurrencies."""
        try:
            response = requests.get(f"{self.BASE_URL}/search/trending")
            data = response.json()
            return data.get('coins', [])
        except Exception as e:
            print(f"CoinGecko error: {e}")
            return []


class MultiSourceDataFetcher:
    """
    Unified interface for fetching data from multiple free sources.
    
    Combines:
    - Alpha Vantage (stocks)
    - News API (sentiment)
    - FRED (economic indicators)
    - CoinGecko (crypto)
    
    Course Concept: Multi-source data integration
    """
    
    def __init__(self, alpha_vantage_key: str = "9BR06FV9HYBIEJQ2",
                 news_api_key: str = None,
                 fred_key: str = None):
        """Initialize all API clients."""
        self.alpha_vantage = AlphaVantageClient(alpha_vantage_key)
        self.news = NewsAPIClient(news_api_key)
        self.fred = FREDClient(fred_key)
        self.crypto = CryptoClient()
    
    def get_comprehensive_view(self, symbol: str) -> Dict:
        """
        Get comprehensive view of a stock with multiple data sources.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with price, indicators, news, and economic context
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Price data
        quote = self.alpha_vantage.get_quote(symbol)
        if quote:
            result['quote'] = quote
        
        # News and sentiment
        news = self.news.get_stock_news(symbol)
        if news:
            result['news'] = news[:5]  # Top 5 articles
            result['news_count'] = len(news)
        
        # Economic context
        vix = self.fred.get_series('VIXCLS')
        if not vix.empty:
            result['vix'] = float(vix.iloc[-1, 0])
        
        return result
    
    def get_market_overview(self) -> Dict:
        """Get broad market overview."""
        return {
            'economic': self.fred.get_economic_context(),
            'crypto_trending': self.crypto.get_trending()[:5],
            'timestamp': datetime.now().isoformat()
        }


def demo_free_apis():
    """Demonstrate free API integration."""
    print("Free API Integration Demo")
    
    # Alpha Vantage demo (with 'demo' key, limited functionality)
    print("\nAlpha Vantage (My Key)...")
    av = AlphaVantageClient("9BR06FV9HYBIEJQ2")
    quote = av.get_quote("IBM")  # Demo key only works with IBM
    print(f"   IBM Quote: {quote}")
    
    # CoinGecko demo (no key required)
    print("\n₿ CoinGecko (No Key Required)...")
    crypto = CryptoClient()
    prices = crypto.get_price(['bitcoin', 'ethereum'])
    print(f"   Bitcoin: ${prices.get('bitcoin', {}).get('usd', 'N/A')}")
    print(f"   Ethereum: ${prices.get('ethereum', {}).get('usd', 'N/A')}")
    
    # Mock news
    print("\n📰 News API (Mock Data)...")
    news = NewsAPIClient()
    articles = news.get_stock_news("AAPL")
    print(f"   Articles found: {len(articles)}")
    
    print("\nAPI demo completed!")


if __name__ == "__main__":
    demo_free_apis()
