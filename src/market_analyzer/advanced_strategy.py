"""
Advanced trading strategies using multiple technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict
from .ml_models import MLStrategy
import talib

class AdvancedTechnicalStrategy(MLStrategy):
    """Trading strategy combining multiple technical indicators (TA-Lib)."""
    
    def __init__(self, params: Dict = None):
        """
        Initialize strategy with parameters.
        """
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
            'volume_ma_period': 20,

            # Additional TA-Lib parameters
            'cci_period': 20,
            'atr_period': 14,
            'stoch_k_period': 14,
            'stoch_d_period': 3
        }
        super().__init__('advanced_technical')
        self.params = {**default_params, **(params or {})}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals with extended TA‑Lib indicators.
        """
        # Initialize signals as float-based Series
        signals = pd.Series(0.0, index=data.index)
        
        try:
            # Basic safety-checks
            if any(col not in data.columns for col in ['Open','High','Low','Close']):
                # Return zero signals if essential columns are missing
                return signals

            # =============== TA-Lib Indicators ===============

            # 1) RSI (fallback to default if missing or user-specified in data)
            close_prices = data['Close'].values
            rsi = talib.RSI(close_prices, timeperiod=self.params['rsi_period'])

            # 2) MACD
            macd, macd_signal, _ = talib.MACD(
                close_prices,
                fastperiod=self.params['macd_fast'],
                slowperiod=self.params['macd_slow'],
                signalperiod=self.params['macd_signal']
            )
            
            # 3) ATR for volatility
            atr = talib.ATR(
                data['High'].values,
                data['Low'].values,
                close_prices,
                timeperiod=self.params['atr_period']
            )

            # 4) CCI
            cci = talib.CCI(
                data['High'].values,
                data['Low'].values,
                close_prices,
                timeperiod=self.params['cci_period']
            )

            # 5) Stochastic (K, D)
            stoch_k, stoch_d = talib.STOCH(
                data['High'].values,
                data['Low'].values,
                close_prices,
                fastk_period=self.params['stoch_k_period'],
                slowk_period=self.params['stoch_d_period'],
                slowd_period=self.params['stoch_d_period']  # You can tweak as needed
            )

            # Convert arrays to Pandas Series for easy alignment
            rsi_s     = pd.Series(rsi, index=data.index)
            macd_s    = pd.Series(macd, index=data.index)
            macd_sig  = pd.Series(macd_signal, index=data.index)
            atr_s     = pd.Series(atr, index=data.index)
            cci_s     = pd.Series(cci, index=data.index)
            stoch_k_s = pd.Series(stoch_k, index=data.index)
            stoch_d_s = pd.Series(stoch_d, index=data.index)

            # ========== Example Basic Rules (You can customize) ==========

            # Buy signal if:
            #   1. RSI < oversold
            #   2. MACD > MACD signal
            #   3. CCI < -100
            #   4. Stoch K crosses above Stoch D
            #   5. ATR is not extremely high (avoid extremely volatile conditions)
            # 
            buy_signals = (
                (rsi_s < self.params['rsi_oversold']) &
                (macd_s > macd_sig) &
                (cci_s < -100) &
                (stoch_k_s > stoch_d_s) &
                (atr_s < atr_s.rolling(10).mean())  # just an example
            )
            
            # Sell signal if:
            #   1. RSI > overbought
            #   2. MACD < MACD signal
            #   3. CCI > +100
            #   4. Stoch K crosses below Stoch D
            # 
            sell_signals = (
                (rsi_s > self.params['rsi_overbought']) &
                (macd_s < macd_sig) &
                (cci_s > 100) &
                (stoch_k_s < stoch_d_s)
            )
            
            # Assign base signals
            signals[buy_signals]  =  1.0
            signals[sell_signals] = -1.0

            # Example: Some position sizing logic
            # Increase position size if RSI is extremely low (< 20) or Stoch is extremely oversold
            # Decrease/invert if extremely high
            conviction = np.zeros(len(signals), dtype=float)
            conviction[ buy_signals & (rsi_s < 20) ] += 0.2
            conviction[ buy_signals & (stoch_k_s < 20) ] += 0.1

            conviction[ sell_signals & (rsi_s > 80) ] -= 0.2
            conviction[ sell_signals & (stoch_k_s > 80) ] -= 0.1
            
            # Scale by (1 + conviction) so we get fractional scaling
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
            strategy_returns = signals.shift(1).fillna(0.0) * price_returns

            # Example 2% stop-loss
            stop_loss = -0.02
            strategy_returns = strategy_returns.clip(lower=stop_loss)
            
            return strategy_returns
        except Exception as e:
            print(f"Error calculating returns: {str(e)}")
            return pd.Series(0.0, index=data.index)
