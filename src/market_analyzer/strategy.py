"""
Trading strategy implementation module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ta

class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (1: Buy, 0: Hold, -1: Sell)."""
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare data for strategy."""
        if data is None or len(data) == 0:
            raise ValueError("Empty data provided")
            
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                raise ValueError("Index must be convertible to datetime")
        
        return data
        
    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns based on signals."""
        data = self._validate_data(data)
        price_returns = data['Close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        return strategy_returns.fillna(0)

class MovingAverageCrossStrategy(TradingStrategy):
    """Moving Average Crossover Strategy."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(f"MA_Cross_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = self._validate_data(data)
        close_prices = data['Close']
        
        # Calculate moving averages
        short_ma = close_prices.rolling(window=self.short_window).mean()
        long_ma = close_prices.rolling(window=self.long_window).mean()
        
        # Generate signals using numpy comparison
        signals = pd.Series(0, index=data.index)
        buy_signals = (short_ma - long_ma) > 0
        sell_signals = (short_ma - long_ma) < 0
        
        signals.loc[buy_signals] = 1
        signals.loc[sell_signals] = -1
        
        return signals.fillna(0)

class RSIStrategy(TradingStrategy):
    """RSI Strategy with oversold/overbought levels."""
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = self._validate_data(data)
        close_prices = data['Close']
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(close_prices, window=self.period).rsi()
        
        # Generate signals using numpy comparison
        signals = pd.Series(0, index=data.index)
        buy_signals = rsi < self.oversold
        sell_signals = rsi > self.overbought
        
        signals.loc[buy_signals] = 1
        signals.loc[sell_signals] = -1
        
        return signals.fillna(0)

class MACDStrategy(TradingStrategy):
    """MACD Strategy."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = self._validate_data(data)
        close_prices = data['Close']
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            close_prices,
            window_slow=self.slow_period,
            window_fast=self.fast_period,
            window_sign=self.signal_period
        )
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        
        # Generate signals using numpy comparison
        signals = pd.Series(0, index=data.index)
        buy_signals = (macd_line - signal_line) > 0
        sell_signals = (macd_line - signal_line) < 0
        
        signals.loc[buy_signals] = 1
        signals.loc[sell_signals] = -1
        
        return signals.fillna(0)

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Strategy."""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"BB_{window}_{num_std}")
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = self._validate_data(data)
        close_prices = data['Close']
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close_prices,
            window=self.window,
            window_dev=self.num_std
        )
        
        # Generate signals using numpy comparison
        signals = pd.Series(0, index=data.index)
        buy_signals = close_prices < bb.bollinger_lband()
        sell_signals = close_prices > bb.bollinger_hband()
        
        signals.loc[buy_signals] = 1
        signals.loc[sell_signals] = -1
        
        return signals.fillna(0)