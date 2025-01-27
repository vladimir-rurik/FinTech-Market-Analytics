"""
Advanced trading strategies using multiple technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .ml_models import MLStrategy

class AdvancedTechnicalStrategy(MLStrategy):
    """Trading strategy combining multiple technical indicators."""
    
    def __init__(self, params: Dict = None):
        """Initialize strategy with parameters."""
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'bb_period': 20,
            'bb_std': 2,
            'volume_ma_period': 20
        }
        super().__init__('advanced_technical')
        self.params = {**default_params, **(params or {})}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using multiple indicators.
        
        Strategy rules:
        1. Trend confirmation using ADX and moving averages
        2. Entry signals from RSI and MACD
        3. Confirmation from volume indicators
        4. Pattern recognition for additional signals
        5. Risk management using volatility indicators
        """
        signals = pd.Series(0, index=data.index)
        
        try:
            # 1. Trend Analysis
            sma_short = data[f'sma_20']
            sma_long = data[f'sma_50']
            adx = data['adx']
            
            trend_up = (sma_short > sma_long) & (adx > self.params['adx_threshold'])
            trend_down = (sma_short < sma_long) & (adx > self.params['adx_threshold'])
            
            # 2. Entry/Exit Signals
            rsi = data[f'rsi_{self.params["rsi_period"]}']
            macd = data['macd']
            macd_signal = data['macd_signal']
            
            # 3. Volume Confirmation
            volume_trend = data['price_vol_corr'] > 0
            
            # 4. Pattern Recognition
            reversal_up = data['hammer'].astype(bool)
            reversal_down = data['shooting_star'].astype(bool)
            
            # 5. Volatility Checks
            bb_squeeze = data[f'bb_width_{self.params["bb_period"]}'] < \
                        data[f'bb_width_{self.params["bb_period"]}'].rolling(20).mean()
            
            # Combine Signals
            
            # Buy Signals:
            # - Uptrend confirmed by ADX and moving averages
            # - RSI oversold or MACD crossing up
            # - Volume confirming
            # - Bullish pattern or volatility breakout
            buy_signals = (
                trend_up &
                ((rsi < self.params['rsi_oversold']) | 
                 (macd > macd_signal)) &
                volume_trend &
                (reversal_up | bb_squeeze)
            )
            
            # Sell Signals:
            # - Downtrend confirmed by ADX and moving averages
            # - RSI overbought or MACD crossing down
            # - Volume confirming
            # - Bearish pattern or volatility breakout
            sell_signals = (
                trend_down &
                ((rsi > self.params['rsi_overbought']) | 
                 (macd < macd_signal)) &
                volume_trend &
                (reversal_down | bb_squeeze)
            )
            
            # Set signals
            signals[buy_signals] = 1
            signals[sell_signals] = -1
            
            # Add position sizing based on conviction
            conviction = np.zeros_like(signals)
            
            # Stronger signals when multiple conditions align
            conviction[buy_signals & (rsi < 20)] += 0.2
            conviction[buy_signals & reversal_up] += 0.2
            conviction[buy_signals & bb_squeeze] += 0.1
            
            conviction[sell_signals & (rsi > 80)] -= 0.2
            conviction[sell_signals & reversal_down] -= 0.2
            conviction[sell_signals & bb_squeeze] -= 0.1
            
            # Scale signals by conviction
            signals = signals * (1 + conviction)
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return pd.Series(0, index=data.index)
        
        return signals

    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Calculate strategy returns with position sizing.
        
        Args:
            data: Market data
            signals: Trading signals with conviction
        
        Returns:
            Strategy returns
        """
        # Calculate base returns
        price_returns = data['Close'].pct_change()
        
        # Apply position sizing based on signal strength
        strategy_returns = signals.shift(1) * price_returns
        
        # Apply stop-loss
        stop_loss = -0.02  # 2% stop loss
        strategy_returns[strategy_returns < stop_loss] = stop_loss
        
        return strategy_returns