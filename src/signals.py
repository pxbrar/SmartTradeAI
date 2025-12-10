"""
SmartTrade AI - Trading Signals

Combines all model outputs to generate final trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class SignalGenerator:
    """Generate trading signals by combining multiple indicators and models."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'technical': 0.3,
            'ml_regression': 0.25,
            'ml_classification': 0.25,
            'sentiment': 0.2
        }
    
    def generate_technical_signal(self, row: pd.Series) -> Tuple[str, float]:
        """Generate signal based on technical indicators."""
        score = 0
        
        # RSI signal
        if 'rsi_14' in row:
            if row['rsi_14'] < 30:
                score += 1  # Oversold - bullish
            elif row['rsi_14'] > 70:
                score -= 1  # Overbought - bearish
        
        # MACD signal
        if 'macd_histogram' in row:
            if row['macd_histogram'] > 0:
                score += 0.5
            else:
                score -= 0.5
        
        # Bollinger Band signal
        if 'bb_percent' in row:
            if row['bb_percent'] < 0.2:
                score += 0.5  # Near lower band
            elif row['bb_percent'] > 0.8:
                score -= 0.5  # Near upper band
        
        # Moving average trend
        if 'sma_20' in row and 'sma_50' in row:
            if row['sma_20'] > row['sma_50']:
                score += 0.5  # Uptrend
            else:
                score -= 0.5  # Downtrend
        
        # Normalize to -1 to 1
        normalized = max(-1, min(1, score / 3))
        
        if normalized > 0.3:
            signal = 'BUY'
        elif normalized < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return signal, normalized
    
    def combine_signals(self, technical_score: float, ml_score: float = 0.0,
                       sentiment_score: float = 0.0) -> Dict:
        """Combine all signals into final recommendation."""
        weights = self.weights
        
        combined = (
            weights['technical'] * technical_score +
            (weights['ml_regression'] + weights['ml_classification']) * ml_score +
            weights['sentiment'] * sentiment_score
        )
        
        if combined > 0.2:
            signal = 'BUY'
            strength = 'STRONG' if combined > 0.5 else 'MODERATE'
        elif combined < -0.2:
            signal = 'SELL'
            strength = 'STRONG' if combined < -0.5 else 'MODERATE'
        else:
            signal = 'HOLD'
            strength = 'NEUTRAL'
        
        return {
            'signal': signal,
            'strength': strength,
            'score': combined,
            'confidence': abs(combined)
        }
    
    def generate_signals_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for entire DataFrame."""
        df = df.copy()
        signals = []
        scores = []
        
        for _, row in df.iterrows():
            signal, score = self.generate_technical_signal(row)
            signals.append(signal)
            scores.append(score)
        
        df['signal'] = signals
        df['signal_score'] = scores
        return df


from typing import Tuple


if __name__ == "__main__":
    print("Signal Generator Test")
    generator = SignalGenerator()
    
    # Test row
    test_row = pd.Series({
        'rsi_14': 25,  # Oversold
        'macd_histogram': 0.5,  # Positive
        'bb_percent': 0.15,  # Near lower band
        'sma_20': 105,
        'sma_50': 100  # Uptrend
    })
    
    signal, score = generator.generate_technical_signal(test_row)
    print(f"Signal: {signal}, Score: {score:.2f}")
    
    combined = generator.combine_signals(score, ml_score=0.3, sentiment_score=0.2)
    print(f"Combined: {combined}")
