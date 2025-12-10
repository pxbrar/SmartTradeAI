"""
SmartTrade AI - LLM Trading Predictor

Uses FREE Large Language Models to analyze stocks and provide buy/sell/hold predictions:
- Google Gemini (Free tier: 15 RPM, 1M tokens/month)
- Ollama (Completely free, runs locally)
- OpenAI (Free trial credits)

Each LLM analyzes:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price action and trends
- Market sentiment
- Risk assessment

Returns structured predictions with confidence levels and reasoning.

Course Concepts Applied:
- Generative AI for financial analysis
- LLM prompt engineering
- Multi-model comparison
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import warnings
warnings.filterwarnings('ignore')

# Try importing LLM libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMTradingPredictor:
    """
    LLM-powered trading signal generator using FREE APIs.
    
    Supported providers (all have free tiers):
    1. Google Gemini - 15 requests/minute, 1M tokens/month FREE
    2. Ollama - Completely FREE, runs locally
    3. OpenAI - Free trial credits
    
    Usage:
        predictor = LLMTradingPredictor(provider='gemini', api_key='AIzaSyD_1m8HSeWnFYjZz8BsXxHCtBqvUtAWs48')
        prediction = predictor.predict('NVDA', indicators, price_data)
    """
    
    # System prompt for trading analysis
    SYSTEM_PROMPT = """You are an expert financial analyst and trading advisor. 
Analyze the provided stock data and technical indicators to give a trading recommendation.

You MUST respond in valid JSON format with this exact structure:
{
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation for the recommendation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "target_price": null or estimated price,
    "stop_loss": null or suggested stop loss price,
    "time_horizon": "SHORT" or "MEDIUM" or "LONG"
}

Be objective and data-driven. Consider both bullish and bearish signals.
Do NOT give financial advice - this is for educational purposes only."""

    def __init__(self, provider: str = 'gemini', api_key: str = None):
        """
        Initialize LLM Trading Predictor.
        
        Args:
            provider: 'gemini', 'ollama', or 'openai'
            api_key: API key for the provider (not needed for Ollama)
            
        Free API Keys:
        - Gemini: https://makersuite.google.com/app/apikey
        - OpenAI: https://platform.openai.com/api-keys (free trial)
        - Ollama: No key needed (local)
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(f'{provider.upper()}_API_KEY')
        self.client = None
        self.model = None
        
        self._setup_client()
    
    def _setup_client(self):
        """Setup the LLM client based on provider."""
        if self.provider == 'gemini':
            if not GEMINI_AVAILABLE:
                print("google-generativeai not installed. Run: pip install google-generativeai")
                return
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')  # Free model
                print("Gemini initialized (gemini-1.5-flash)")
            else:
                print("No Gemini API key. Get free key at: https://makersuite.google.com/app/apikey")
        

        
        elif self.provider == 'ollama':
            # Ollama runs locally - no API key needed
            self.model = "mistral"  # or llama2, codellama, etc.
            print("Ollama initialized (local). Make sure Ollama is running!")
        
        elif self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                print("openai not installed. Run: pip install openai")
                return
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                self.model = "gpt-3.5-turbo"
                print("OpenAI initialized (gpt-3.5-turbo)")
            else:
                print("No OpenAI API key.")
        
        else:
            print(f"Unknown provider: {self.provider}")
    
    def _format_analysis_prompt(self, symbol: str, indicators: Dict, 
                                 price_data: pd.DataFrame = None) -> str:
        """Format the analysis prompt with stock data."""
        prompt = f"""
# Stock Analysis Request: {symbol}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Technical Indicators
"""
        # Add indicators
        for name, value in indicators.items():
            if isinstance(value, (int, float)):
                prompt += f"- {name}: {value:.4f}\n"
            else:
                prompt += f"- {name}: {value}\n"
        
        # Add interpretation guidelines
        prompt += """
## Indicator Interpretation Guide
- RSI > 70: Overbought (bearish signal)
- RSI < 30: Oversold (bullish signal)
- MACD > Signal: Bullish crossover
- MACD < Signal: Bearish crossover
- Price > Upper BB: Overbought
- Price < Lower BB: Oversold
- Positive momentum: Bullish
- Negative momentum: Bearish

## Price Context
"""
        if price_data is not None and len(price_data) > 0:
            latest = price_data.iloc[-1]
            prompt += f"- Current Price: ${latest.get('close', 'N/A'):.2f}\n"
            
            if len(price_data) > 5:
                week_ago = price_data.iloc[-5]
                weekly_change = (latest['close'] / week_ago['close'] - 1) * 100
                prompt += f"- 5-Day Change: {weekly_change:+.2f}%\n"
            
            if len(price_data) > 20:
                month_ago = price_data.iloc[-20]
                monthly_change = (latest['close'] / month_ago['close'] - 1) * 100
                prompt += f"- 20-Day Change: {monthly_change:+.2f}%\n"
            
            if 'volume' in latest:
                prompt += f"- Volume: {latest['volume']:,.0f}\n"
        
        prompt += """
## Task
Based on the above technical analysis, provide a BUY, SELL, or HOLD recommendation.
Respond ONLY with valid JSON in the specified format.
"""
        return prompt
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        if not self.model:
            return json.dumps({"error": "Gemini not configured"})
        
        try:
            response = self.model.generate_content(
                f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1000,
                }
            )
            return response.text
        except Exception as e:
            return json.dumps({"error": str(e)})
    

    
    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama server."""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=240  # 4 minute timeout for slow model loads
            )
            if response.status_code == 200:
                return response.json()['response']
            else:
                return json.dumps({"error": f"Ollama error: {response.status_code}"})
        except requests.exceptions.ConnectionError:
            return json.dumps({"error": "Ollama not running. Start with: ollama serve"})
        except requests.exceptions.Timeout:
            return json.dumps({"error": "Ollama timed out. Model may be loading - try again in a moment."})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not self.client:
            return json.dumps({"error": "OpenAI not configured"})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response into structured format."""
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing fails, create a basic response
        signal = "HOLD"
        if "buy" in response.lower():
            signal = "BUY"
        elif "sell" in response.lower():
            signal = "SELL"
        
        return {
            "signal": signal,
            "confidence": 0.5,
            "reasoning": response[:500],
            "key_factors": [],
            "risk_level": "MEDIUM",
            "target_price": None,
            "stop_loss": None,
            "time_horizon": "MEDIUM",
            "raw_response": response
        }
    
    def predict(self, symbol: str, indicators: Dict, 
                price_data: pd.DataFrame = None) -> Dict:
        """
        Get LLM trading prediction for a stock.
        
        Args:
            symbol: Stock ticker (e.g., 'NVDA')
            indicators: Dict of technical indicators
            price_data: Optional DataFrame with price history
            
        Returns:
            Dict with signal, confidence, reasoning, etc.
        """
        prompt = self._format_analysis_prompt(symbol, indicators, price_data)
        
        # Call appropriate LLM
        if self.provider == 'gemini':
            response = self._call_gemini(prompt)

        elif self.provider == 'ollama':
            response = self._call_ollama(prompt)
        elif self.provider == 'openai':
            response = self._call_openai(prompt)
        else:
            response = json.dumps({"error": f"Unknown provider: {self.provider}"})
        
        # Parse response
        result = self._parse_response(response)
        result['symbol'] = symbol
        result['provider'] = self.provider
        result['timestamp'] = datetime.now().isoformat()
        
        return result
    
    def predict_batch(self, stocks: Dict[str, Dict]) -> List[Dict]:
        """
        Get predictions for multiple stocks.
        
        Args:
            stocks: Dict of {symbol: {indicators: {...}, price_data: df}}
            
        Returns:
            List of predictions
        """
        predictions = []
        for symbol, data in stocks.items():
            pred = self.predict(
                symbol, 
                data.get('indicators', {}),
                data.get('price_data')
            )
            predictions.append(pred)
        return predictions
    
    def explain_signal(self, prediction: Dict) -> str:
        """Generate a human-readable explanation of the prediction."""
        signal = prediction.get('signal', 'HOLD')
        confidence = prediction.get('confidence', 0.5)
        reasoning = prediction.get('reasoning', 'No reasoning provided')
        risk = prediction.get('risk_level', 'MEDIUM')
        factors = prediction.get('key_factors', [])
        
        emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(signal, '⚪')
        
        explanation = f"""
{emoji} **{signal}** Signal for {prediction.get('symbol', 'Unknown')}

**Confidence:** {confidence*100:.0f}%
**Risk Level:** {risk}
**Provider:** {prediction.get('provider', 'Unknown')}

**Reasoning:**
{reasoning}
"""
        if factors:
            explanation += "\n**Key Factors:**\n"
            for factor in factors:
                explanation += f"  • {factor}\n"
        
        if prediction.get('target_price'):
            explanation += f"\n**Target Price:** ${prediction['target_price']:.2f}"
        if prediction.get('stop_loss'):
            explanation += f"\n**Stop Loss:** ${prediction['stop_loss']:.2f}"
        
        return explanation


class MultiLLMPredictor:
    """
    Combine predictions from multiple LLMs for more robust signals.
    
    Uses ensemble voting across different models to reduce individual
    model biases and increase prediction reliability.
    """
    
    def __init__(self, providers: List[Dict] = None):
        """
        Initialize with multiple LLM providers.
        
        Args:
            providers: List of {provider: str, api_key: str} dicts
        """
        self.predictors = []
        
        if providers:
            for p in providers:
                try:
                    predictor = LLMTradingPredictor(
                        provider=p['provider'],
                        api_key=p.get('api_key')
                    )
                    self.predictors.append(predictor)
                except Exception as e:
                    print(f"Failed to initialize {p['provider']}: {e}")
    
    def predict(self, symbol: str, indicators: Dict, 
                price_data: pd.DataFrame = None) -> Dict:
        """
        Get ensemble prediction from multiple LLMs.
        
        Returns aggregated prediction with voting results.
        """
        predictions = []
        
        for predictor in self.predictors:
            try:
                pred = predictor.predict(symbol, indicators, price_data)
                predictions.append(pred)
            except Exception as e:
                print(f"Error with {predictor.provider}: {e}")
        
        if not predictions:
            return {"error": "No predictions available", "signal": "HOLD"}
        
        # Aggregate predictions
        signals = [p.get('signal', 'HOLD') for p in predictions]
        confidences = [p.get('confidence', 0.5) for p in predictions]
        
        # Voting
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        hold_votes = signals.count('HOLD')
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus_signal = 'BUY'
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus_signal = 'SELL'
        else:
            consensus_signal = 'HOLD'
        
        # Agreement level
        max_votes = max(buy_votes, sell_votes, hold_votes)
        agreement = max_votes / len(predictions)
        
        return {
            'symbol': symbol,
            'consensus_signal': consensus_signal,
            'agreement_level': agreement,
            'average_confidence': np.mean(confidences),
            'votes': {'BUY': buy_votes, 'SELL': sell_votes, 'HOLD': hold_votes},
            'individual_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }


def demo_llm_predictor():
    """Demonstrate LLM trading prediction with real stock data."""
    print("🤖 LLM Trading Predictor Demo")
    print("=" * 50)
    
    # Import data collection and indicators to use real data
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_collection import StockDataCollector
    from src.indicators import calculate_all_indicators
    
    # Fetch real NVDA data
    print("\n📈 Fetching real NVDA data from Yahoo Finance...")
    collector = StockDataCollector()
    df = collector.fetch_stock_data('NVDA', period='1y')
    
    if df.empty:
        print("❌ Failed to fetch data. Check internet connection.")
        return
    
    # Calculate all technical indicators
    df = calculate_all_indicators(df)
    print(f"✅ Loaded {len(df)} rows of real NVDA data")
    
    # Extract latest indicators from real data
    latest = df.iloc[-1]
    real_indicators = {}
    indicator_cols = ['rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                      'bb_upper', 'bb_lower', 'bb_percent', 'sma_20', 
                      'sma_50', 'momentum_10', 'volatility_20', 'close']
    
    for col in indicator_cols:
        if col in df.columns and pd.notna(latest[col]):
            real_indicators[col] = float(latest[col])
    
    # Get last 20 days of price data
    price_data = df[['close', 'volume']].tail(20).copy()
    
    print("\n📊 Real Technical Indicators:")
    for name, value in real_indicators.items():
        print(f"   {name}: {value:.4f}")
    
    # Try Ollama first (free, local)
    print("\n🔄 Attempting Ollama (local LLM)...")
    predictor = LLMTradingPredictor(provider='ollama')
    prediction = predictor.predict('NVDA', real_indicators, price_data)
    
    if 'error' in prediction:
        print(f"LLM Error: {prediction.get('error', 'Unknown')}")
        print("\nTo enable LLM predictions:")
        print("  1. Install Ollama: brew install ollama")
        print("  2. Start server: ollama serve")
        print("  3. Pull a model: ollama pull mistral")
        print("  OR")
        print("  4. Set GEMINI_API_KEY in .env file")
    else:
        print("\n" + predictor.explain_signal(prediction))


def get_llm_prediction(symbol: str, df: pd.DataFrame, 
                        provider: str = 'gemini', 
                        api_key: str = None) -> Dict:
    """
    Convenience function to get LLM prediction for a stock.
    
    Args:
        symbol: Stock ticker
        df: DataFrame with processed stock data (must have indicators)
        provider: LLM provider to use
        api_key: API key for the provider
        
    Returns:
        Dict with prediction details
    """
    # Extract latest indicators from DataFrame
    latest = df.iloc[-1]
    
    indicators = {}
    indicator_cols = ['rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                      'bb_upper', 'bb_lower', 'bb_percent', 'sma_20', 
                      'sma_50', 'momentum_10', 'volatility_20', 'stoch_k',
                      'stoch_d', 'atr', 'obv', 'daily_return']
    
    for col in indicator_cols:
        if col in df.columns:
            indicators[col] = float(latest[col])
    
    # Get last 20 days of price data
    price_data = df[['close', 'volume']].tail(20).copy()
    
    # Create predictor and get prediction
    predictor = LLMTradingPredictor(provider=provider, api_key=api_key)
    return predictor.predict(symbol, indicators, price_data)


if __name__ == "__main__":
    demo_llm_predictor()
