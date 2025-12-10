"""
Sentiment Analysis for Stock News

Analyzes news headlines to figure out if the sentiment is 
positive, negative, or neutral. Uses TextBlob + keyword matching.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Analyze sentiment of financial news headlines."""
    
    def __init__(self):
        # words that usually mean good news for stocks
        self.positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'strong',
            'beat', 'exceed', 'upgrade', 'buy', 'outperform', 'boom', 'soar'
        ]
        # words that usually mean bad news
        self.negative_words = [
            'bearish', 'crash', 'drop', 'loss', 'decline', 'weak', 'miss',
            'downgrade', 'sell', 'underperform', 'bust', 'plunge', 'fall'
        ]
    
    def clean_text(self, text):
        """Clean up the text - remove URLs and special chars."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', '', text)  # remove links
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # only keep letters
        return text.lower().strip()
    
    def analyze_sentiment(self, text):
        """Figure out if text is positive, negative, or neutral."""
        clean = self.clean_text(text)
        blob = TextBlob(clean)
        
        # TextBlob gives us polarity from -1 to 1
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # also count our financial keywords
        positive_count = sum(1 for word in self.positive_words if word in clean)
        negative_count = sum(1 for word in self.negative_words if word in clean)
        keyword_score = (positive_count - negative_count) / max(1, positive_count + negative_count)
        
        # combine both scores (textblob + keywords)
        combined_score = 0.7 * polarity + 0.3 * keyword_score
        
        # decide the label
        if combined_score > 0.1:
            label = 'positive'
        elif combined_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'keyword_score': keyword_score,
            'combined_score': combined_score,
            'label': label
        }
    
    def analyze_headlines(self, headlines):
        """Analyze a bunch of headlines at once."""
        results = []
        for headline in headlines:
            sentiment = self.analyze_sentiment(headline)
            sentiment['headline'] = headline
            results.append(sentiment)
        return pd.DataFrame(results)
    
    def get_aggregate_sentiment(self, headlines):
        """Get overall sentiment from multiple headlines."""
        df = self.analyze_headlines(headlines)
        return {
            'avg_score': df['combined_score'].mean(),
            'positive_pct': (df['label'] == 'positive').mean() * 100,
            'negative_pct': (df['label'] == 'negative').mean() * 100,
            'neutral_pct': (df['label'] == 'neutral').mean() * 100,
            'count': len(df)
        }


class RealTimeSentiment:
    """Fetch real headlines from free APIs."""
    
    def __init__(self):
        import os
        self.finnhub_key = os.environ.get('FINNHUB_API_KEY', '')
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY', '')
        self._cache = {}
        self._cache_ttl = 300  # cache for 5 minutes
        self._cache_time = {}
    
    def _is_cache_valid(self, symbol):
        """Check if we still have fresh cached data."""
        import time
        if symbol not in self._cache_time:
            return False
        return (time.time() - self._cache_time[symbol]) < self._cache_ttl
    
    def _fetch_yahoo_rss(self, symbol):
        """Get headlines from Yahoo Finance RSS (free, no API key)."""
        import urllib.request
        import xml.etree.ElementTree as ET
        
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read()
            
            root = ET.fromstring(content)
            headlines = []
            for item in root.findall('.//item'):
                title = item.find('title')
                if title is not None and title.text:
                    headlines.append(title.text)
            
            if headlines:
                return headlines[:10], 'Yahoo Finance'
        except Exception as e:
            print(f"Yahoo RSS error: {e}")
        
        return [], ''
    
    def _fetch_finnhub(self, symbol):
        """Get headlines from Finnhub (need API key, 60 calls/min free)."""
        import urllib.request
        import json
        from datetime import datetime, timedelta
        
        if not self.finnhub_key:
            return [], ''
        
        try:
            today = datetime.now()
            from_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={self.finnhub_key}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read())
            
            headlines = [item.get('headline', '') for item in data[:10] if item.get('headline')]
            if headlines:
                return headlines, 'Finnhub'
        except Exception as e:
            print(f"Finnhub error: {e}")
        
        return [], ''
    
    def _fetch_alpha_vantage(self, symbol):
        """Get headlines from Alpha Vantage (need API key, 25 calls/day free)."""
        import urllib.request
        import json
        
        if not self.alpha_vantage_key:
            return [], ''
        
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alpha_vantage_key}&limit=10"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read())
            
            feed = data.get('feed', [])
            headlines = [item.get('title', '') for item in feed[:10] if item.get('title')]
            if headlines:
                return headlines, 'Alpha Vantage'
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
        
        return [], ''
    
    def fetch_headlines(self, symbol):
        """Try to get headlines from various sources."""
        import time
        
        # check cache first
        if self._is_cache_valid(symbol):
            return self._cache[symbol]
        
        # try each source until one works
        for fetch_func in [self._fetch_yahoo_rss, self._fetch_finnhub, self._fetch_alpha_vantage]:
            headlines, source = fetch_func(symbol)
            if headlines:
                self._cache[symbol] = (headlines, source)
                self._cache_time[symbol] = time.time()
                return headlines, source
        
        # if nothing works, return empty
        print(f"No news sources available for {symbol}. Configure API keys for real data.")
        return [], 'No Data Available'
    
    def get_realtime_sentiment(self, symbol, analyzer=None):
        """Get sentiment analysis for a stock."""
        if analyzer is None:
            analyzer = SentimentAnalyzer()
        
        headlines, source = self.fetch_headlines(symbol)
        
        # analyze each headline
        analyzed = []
        for h in headlines:
            sentiment = analyzer.analyze_sentiment(h)
            analyzed.append({
                'headline': h[:100],
                'score': sentiment['combined_score'],
                'label': sentiment['label']
            })
        
        agg = analyzer.get_aggregate_sentiment(headlines)
        
        return {
            'avg_score': agg['avg_score'],
            'positive_pct': agg['positive_pct'],
            'negative_pct': agg['negative_pct'],
            'neutral_pct': agg['neutral_pct'],
            'headlines': analyzed,
            'source': source,
            'count': len(headlines)
        }


# test it
if __name__ == "__main__":
    print("Testing Sentiment Analysis...")
    
    analyzer = SentimentAnalyzer()
    
    # Test with manual headlines
    test_headlines = [
        "Company reports strong Q4 earnings",
        "Stock drops on weak guidance",
        "Market remains stable"
    ]
    
    for headline in test_headlines:
        result = analyzer.analyze_sentiment(headline)
        print(f"{result['label'].upper():>8}: {headline[:50]}...")
    
    agg = analyzer.get_aggregate_sentiment(test_headlines)
    print(f"\nOverall: {agg['avg_score']:.2f} ({agg['positive_pct']:.0f}% positive)")
    
    print("\nTesting real-time fetch...")
    rt = RealTimeSentiment()
    result = rt.get_realtime_sentiment("AAPL")
    print(f"Source: {result['source']}, Headlines: {result['count']}")
    print("Done!")
