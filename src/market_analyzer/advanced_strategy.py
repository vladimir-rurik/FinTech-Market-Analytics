"""
Advanced trading strategies using multiple technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict
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
        signals = pd.Series(0.0, index=data.index)  # <-- FLOAT series

        try:
            # Safely get each column; fallback to zeros or neutral values if missing
            sma_short = data.get('sma_20', pd.Series(np.nan, index=data.index))
            sma_long  = data.get('sma_50', pd.Series(np.nan, index=data.index))
            adx       = data.get('adx', pd.Series(0.0, index=data.index))

            # Confirm up/down trend
            trend_up   = (sma_short > sma_long) & (adx > self.params['adx_threshold'])
            trend_down = (sma_short < sma_long) & (adx > self.params['adx_threshold'])
            
            # RSI
            rsi_col = f"rsi_{self.params['rsi_period']}"
            rsi = data.get(rsi_col, pd.Series(50.0, index=data.index))  # fallback to 50

            # MACD & signal
            macd        = data.get('macd', pd.Series(0.0, index=data.index))
            macd_signal = data.get('macd_signal', pd.Series(0.0, index=data.index))
            
            # Volume confirmation
            # If 'price_vol_corr' missing, fallback to True so it doesn't block signals
            price_vol_corr = data.get('price_vol_corr', pd.Series(1.0, index=data.index))
            volume_trend   = price_vol_corr > 0

            # Candlestick patterns
            reversal_up   = data.get('hammer', pd.Series(False, index=data.index)).astype(bool)
            reversal_down = data.get('shooting_star', pd.Series(False, index=data.index)).astype(bool)

            # Bollinger band squeeze
            bb_col = f"bb_width_{self.params['bb_period']}"
            if bb_col in data.columns:
                rolling_mean = data[bb_col].rolling(20).mean()
                bb_squeeze   = data[bb_col] < rolling_mean
            else:
                bb_squeeze   = pd.Series(False, index=data.index)
            # Combine Signals
            
            # Buy Signals:
            # - Uptrend confirmed by ADX and moving averages
            # - RSI oversold or MACD crossing up
            # - Volume confirming
            # - Bullish pattern or volatility breakout
            buy_signals = (
                trend_up
                & ((rsi < self.params['rsi_oversold']) | (macd > macd_signal))
                & volume_trend
                & (reversal_up | bb_squeeze)
            )
            
            # Sell Signals:
            # - Downtrend confirmed by ADX and moving averages
            # - RSI overbought or MACD crossing down
            # - Volume confirming
            # - Bearish pattern or volatility breakout
            sell_signals = (
                trend_down
                & ((rsi > self.params['rsi_overbought']) | (macd < macd_signal))
                & volume_trend
                & (reversal_down | bb_squeeze)
            )

            # Assign base signals as +/- 1.0
            signals[buy_signals]  =  1.0
            signals[sell_signals] = -1.0

            # Add position sizing with a separate float array
            conviction = np.zeros(len(signals), dtype=float)

            # Example extra conditions for scaling
            conviction[ buy_signals & (rsi < 20) ]         += 0.2
            conviction[ buy_signals & reversal_up ]        += 0.2
            conviction[ buy_signals & bb_squeeze ]         += 0.1

            conviction[ sell_signals & (rsi > 80) ]        -= 0.2
            conviction[ sell_signals & reversal_down ]     -= 0.2
            conviction[ sell_signals & bb_squeeze ]        -= 0.1

            # Multiply signals by (1 + conviction) so we get fractional sizes
            signals = signals * (1 + conviction)

        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return pd.Series(0.0, index=data.index)

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
        try:
            price_returns = data['Close'].pct_change().fillna(0.0)
            # Use prior day's signal
            strategy_returns = signals.shift(1).fillna(0.0) * price_returns
            
            # Clip by a stop-loss
            stop_loss = -0.02  # 2%
            strategy_returns = strategy_returns.clip(lower=stop_loss)
            return strategy_returns
        except Exception as e:
            print(f"Error calculating returns: {str(e)}")
            return pd.Series(0.0, index=data.index)
