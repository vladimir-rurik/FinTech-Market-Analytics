"""
Advanced feature engineering for market data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import ta
from scipy import stats

class FeatureEngineer:
    """Feature engineering for market data."""
    
    def __init__(self):
        """Initialize feature engineer with default parameters."""
        self.price_windows = [5, 10, 20, 50, 100]
        self.volatility_windows = [5, 10, 20]
        self.momentum_windows = [5, 10, 20]
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Features:
        - Moving averages and their ratios
        - Price momentum indicators
        - Support and resistance levels
        """
        features = pd.DataFrame(index=data.index)
        
        # Moving averages and their ratios
        for window in self.price_windows:
            features[f'sma_{window}'] = data['Close'].rolling(window=window).mean()
            features[f'ema_{window}'] = data['Close'].ewm(span=window).mean()
            features[f'close_to_sma_{window}'] = data['Close'] / features[f'sma_{window}']
        
        # Price momentum
        for window in self.momentum_windows:
            features[f'momentum_{window}'] = data['Close'].pct_change(window)
            features[f'acceleration_{window}'] = features[f'momentum_{window}'].diff()
        
        # Support and resistance
        for window in self.price_windows:
            features[f'support_{window}'] = data['Low'].rolling(window=window).min()
            features[f'resistance_{window}'] = data['High'].rolling(window=window).max()
            features[f'price_channel_pos_{window}'] = (data['Close'] - features[f'support_{window}']) / \
                (features[f'resistance_{window}'] - features[f'support_{window}'])
        
        return features

    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features.
        
        Features:
        - Rolling volatility
        - Volatility regimes
        - Price ranges and gaps
        """
        features = pd.DataFrame(index=data.index)
        
        # Calculate returns
        returns = data['Close'].pct_change()
        log_returns = np.log1p(data['Close']).diff()
        
        # Rolling volatility
        for window in self.volatility_windows:
            features[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
            features[f'log_volatility_{window}'] = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        # Volatility regimes
        for window in self.volatility_windows:
            vol = features[f'volatility_{window}']
            features[f'vol_regime_{window}'] = pd.qcut(vol, q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Price ranges and gaps
        features['daily_range'] = (data['High'] - data['Low']) / data['Close']
        features['gap_up'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        features['gap_down'] = (data['Open'] - data['High'].shift(1)) / data['High'].shift(1)
        
        return features

    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Features:
        - Volume momentum
        - Price-volume correlations
        - Volume profiles
        """
        features = pd.DataFrame(index=data.index)
        
        # Volume momentum
        for window in self.momentum_windows:
            features[f'volume_ma_{window}'] = data['Volume'].rolling(window=window).mean()
            features[f'volume_ratio_{window}'] = data['Volume'] / features[f'volume_ma_{window}']
            features[f'volume_momentum_{window}'] = data['Volume'].pct_change(window)
        
        # Price-volume correlations
        for window in self.momentum_windows:
            price_changes = data['Close'].pct_change()
            volume_changes = data['Volume'].pct_change()
            features[f'price_volume_corr_{window}'] = \
                price_changes.rolling(window).corr(volume_changes)
        
        # Volume profiles
        features['volume_profile'] = pd.qcut(data['Volume'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        features['relative_volume'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        return features

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features.
        
        Features:
        - RSI variations
        - MACD components
        - Bollinger Bands signals
        """
        features = pd.DataFrame(index=data.index)
        
        # RSI variations
        for window in self.momentum_windows:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=window)
            features[f'rsi_{window}'] = rsi.rsi()
            features[f'rsi_ma_{window}'] = features[f'rsi_{window}'].rolling(window=window).mean()
        
        # MACD variations
        for (fast, slow) in [(12, 26), (5, 35), (8, 21)]:
            macd = ta.trend.MACD(data['Close'], window_fast=fast, window_slow=slow)
            features[f'macd_{fast}_{slow}'] = macd.macd()
            features[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
            features[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()
        
        # Bollinger Bands
        for window in self.price_windows:
            bb = ta.volatility.BollingerBands(data['Close'], window=window)
            features[f'bb_width_{window}'] = bb.bollinger_wband()
            features[f'bb_position_{window}'] = \
                (data['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        return features

    def create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create machine learning specific features.
        
        Features:
        - Target variables
        - Time-based features
        - Statistical features
        """
        features = pd.DataFrame(index=data.index)
        
        # Future returns (target variables)
        for horizon in [1, 5, 10, 20]:
            features[f'future_return_{horizon}'] = data['Close'].pct_change(horizon).shift(-horizon)
        
        # Time-based features
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        
        # Statistical features
        returns = data['Close'].pct_change()
        for window in self.price_windows:
            features[f'returns_skew_{window}'] = returns.rolling(window).skew()
            features[f'returns_kurt_{window}'] = returns.rolling(window).kurt()
            
        return features

    def process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process all features for a given dataset.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        all_features = pd.DataFrame(index=data.index)
        
        # Create feature groups
        price_features = self.create_price_features(data)
        volatility_features = self.create_volatility_features(data)
        volume_features = self.create_volume_features(data)
        technical_features = self.create_technical_features(data)
        ml_features = self.create_ml_features(data)
        
        # Combine all features
        feature_groups = [
            price_features,
            volatility_features,
            volume_features,
            technical_features,
            ml_features
        ]
        
        for group in feature_groups:
            for column in group.columns:
                all_features[column] = group[column]
        
        return all_features