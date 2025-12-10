"""
Time Series Forecasting with ARIMA

ARIMA model to predict future stock prices.
Uses past prices to forecast where price will go.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA model for predicting stock prices.
    ARIMA(p,d,q) - p is lag order, d is differencing, q is moving avg order
    """
    
    def __init__(self, order=(5, 1, 0)):
        self.order = order  # (p, d, q)
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
    
    def check_stationarity(self, series):
        """
        ADF test to check if series is stationary.
        p-value < 0.05 means its stationary (good for ARIMA)
        """
        result = adfuller(series.dropna())
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def fit(self, series):
        """Fit the ARIMA model on historical prices."""
        series = series.dropna()
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        
        # AIC and BIC - lower is better
        return {
            'aic': self.fitted_model.aic, 
            'bic': self.fitted_model.bic, 
            'order': self.order
        }
    
    def forecast(self, steps=30):
        """Predict future prices."""
        if not self.is_fitted:
            raise ValueError("fit the model first!")
        
        fc = self.fitted_model.get_forecast(steps=steps)
        conf = fc.conf_int()
        
        return pd.DataFrame({
            'forecast': fc.predicted_mean.values,
            'lower_ci': conf.iloc[:, 0].values,  # 95% confidence lower bound
            'upper_ci': conf.iloc[:, 1].values   # 95% confidence upper bound
        })
    
    def evaluate(self, series, test_size=30):
        """
        See how good our predictions are.
        Train on older data, test on recent data.
        """
        train, test = series[:-test_size], series[-test_size:]
        
        model = ARIMA(train, order=self.order)
        fitted = model.fit()
        pred = fitted.forecast(steps=test_size)
        
        # calculate error metrics
        rmse = np.sqrt(mean_squared_error(test, pred))
        mae = mean_absolute_error(test, pred)
        mape = np.mean(np.abs((test.values - pred.values) / test.values)) * 100
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape}


def forecast_stock(df, steps=30, order=(5, 1, 0)):
    """Quick way to forecast stock prices."""
    series = df.set_index('date')['close'].sort_index()
    
    forecaster = ARIMAForecaster(order=order)
    fit_metrics = forecaster.fit(series)
    forecast_df = forecaster.forecast(steps=steps)
    eval_metrics = forecaster.evaluate(series, test_size=min(30, len(series)//5))
    
    return {
        'forecaster': forecaster,
        'fit_metrics': fit_metrics,
        'forecast': forecast_df,
        'evaluation': eval_metrics,
        'last_price': series.iloc[-1]
    }


# test it with data
if __name__ == "__main__":
    print("Testing ARIMA Forecasting with Real Stock Data...")
    
    # Import data collection
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_collection import StockDataCollector
    
    # Fetch AAPL data
    collector = StockDataCollector()
    df = collector.fetch_stock_data('AAPL', period='5y')
    
    if df.empty:
        print("Failed to fetch data, check internet connection")
        exit(1)
    
    print(f"Loaded {len(df)} rows of real AAPL data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Run ARIMA forecasting
    result = forecast_stock(df, steps=30, order=(5, 1, 0))
    
    print(f"\nARIMA Model Fit:")
    print(f"  AIC: {result['fit_metrics']['aic']:.2f}")
    print(f"  BIC: {result['fit_metrics']['bic']:.2f}")
    
    print(f"\nEvaluation Metrics:")
    print(f"  RMSE: ${result['evaluation']['rmse']:.2f}")
    print(f"  MAE: ${result['evaluation']['mae']:.2f}")
    print(f"  MAPE: {result['evaluation']['mape']:.2f}%")
    
    print(f"\n30-Day Forecast (from ${result['last_price']:.2f}):")
    forecast = result['forecast']
    print(f"  Day 1: ${forecast['forecast'].iloc[0]:.2f} (${forecast['lower_ci'].iloc[0]:.2f} - ${forecast['upper_ci'].iloc[0]:.2f})")
    print(f"  Day 30: ${forecast['forecast'].iloc[-1]:.2f} (${forecast['lower_ci'].iloc[-1]:.2f} - ${forecast['upper_ci'].iloc[-1]:.2f})")
    
    print("Done!")

