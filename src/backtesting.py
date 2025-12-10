"""
SmartTrade AI - Backtesting Engine

Backtest trading strategies on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    signal: str
    pnl: float
    pnl_pct: float


class Backtester:
    """Backtest trading strategies."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
    
    def run_backtest(self, df: pd.DataFrame, signal_col: str = 'signal') -> Dict:
        """Run backtest on signals."""
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        self.trades = []
        equity = [capital]
        
        for i, row in df.iterrows():
            current_price = row['close']
            signal = row.get(signal_col, 'HOLD')
            
            # Entry
            if position == 0 and signal == 'BUY':
                position = capital / current_price
                entry_price = current_price
                entry_date = row['date']
            
            # Exit
            elif position > 0 and signal == 'SELL':
                exit_price = current_price
                pnl = position * (exit_price - entry_price)
                pnl_pct = (exit_price / entry_price - 1) * 100
                
                self.trades.append(Trade(
                    entry_date=str(entry_date),
                    exit_date=str(row['date']),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    signal='LONG',
                    pnl=pnl,
                    pnl_pct=pnl_pct
                ))
                
                capital += pnl
                position = 0
            
            # Update equity
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity.append(current_equity)
        
        self.equity_curve = equity
        return self._calculate_metrics(df)
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        final_equity = self.equity_curve[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Trade statistics
        pnls = [t.pnl for t in self.trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Sharpe ratio (simplified)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max drawdown
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        return pd.DataFrame({'equity': self.equity_curve})


def run_simple_backtest(df: pd.DataFrame, signal_col: str = 'signal',
                        initial_capital: float = 100000) -> Dict:
    """Convenience function to run backtest."""
    bt = Backtester(initial_capital)
    return bt.run_backtest(df, signal_col)


if __name__ == "__main__":
    print("Backtesting Engine Test with Real Data")
    
    # Import data collection and indicators
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from src.data_collection import StockDataCollector
    from src.indicators import calculate_all_indicators
    from src.signals import SignalGenerator
    
    # Fetch AAPL data
    collector = StockDataCollector()
    df = collector.fetch_stock_data('AAPL', period='2y')
    
    if df.empty:
        print("Failed to fetch data, check internet connection")
        exit(1)
    
    # Calculate indicators
    df = calculate_all_indicators(df)
    print(f"Loaded {len(df)} rows of real AAPL data")
    
    # Generate signals using SignalGenerator
    generator = SignalGenerator()
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        signal, _ = generator.generate_technical_signal(row)
        signals.append(signal)
    df['signal'] = signals
    
    # Run backtest
    bt = Backtester(100000)
    results = bt.run_backtest(df)
    
    print(f"\nBacktest Results for AAPL:")
    print(f"  Total Return: {results.get('total_return_pct', 0):.2f}%")
    print(f"  Win Rate: {results.get('win_rate_pct', 0):.1f}%")
    print(f"  Total Trades: {results.get('total_trades', 0)}")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")

