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
    
    def _ensure_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all price and volume data is numeric."""
        data = data.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data
    
    def create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic price and volume features."""
        data = self._ensure_numeric(data)
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log1p(data['Close']).diff()
        features['price_range'] = (data['High'] - data['Low']) / data['Close']
        
        # Volume features
        if 'Volume' in data.columns:
            features['volume_change'] = data['Volume'].pct_change()
            features['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        return features
    
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
        data = self._ensure_numeric(data)
        features = pd.DataFrame(index=data.index)
        
        try:
            # Basic volatility
            returns = data['Close'].pct_change()
            
            for window in self.volatility_windows:
                # Standard volatility
                features[f'volatility_{window}'] = returns.rolling(
                    window=window).std() * np.sqrt(252)
                
                # Parkinson volatility (uses High-Low range)
                high_low_range = np.log(data['High'] / data['Low'])
                features[f'parkison_vol_{window}'] = np.sqrt(
                    1 / (4 * np.log(2)) * high_low_range.rolling(
                        window=window).mean() * 252)
            
        except Exception as e:
            print(f"Error in volatility features: {str(e)}")
        
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
        data = self._ensure_numeric(data)
        features = pd.DataFrame(index=data.index)
        
        try:
            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f'sma_{window}'] = data['Close'].rolling(window=window).mean()
                features[f'ema_{window}'] = data['Close'].ewm(span=window).mean()
            
            # RSI
            rsi = ta.momentum.RSIIndicator(data['Close'])
            features['rsi'] = rsi.rsi()
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            features['bb_high'] = bb.bollinger_hband()
            features['bb_low'] = bb.bollinger_lband()
            features['bb_width'] = bb.bollinger_wband()
            
        except Exception as e:
            print(f"Error in technical features: {str(e)}")
        
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
        try:
            # Create feature groups
            all_features = pd.DataFrame(index=data.index)
            
            feature_groups = [
                self.create_basic_features(data),
                self.create_technical_features(data),
                self.create_volatility_features(data)
            ]
            
            # Combine features
            for group in feature_groups:
                if not group.empty:
                    for col in group.columns:
                        all_features[col] = group[col]
            
            # Clean up features
            all_features = all_features.astype(float)
            all_features = all_features.ffill().bfill().fillna(0)
            
            print(f"Created {len(all_features.columns)} features")
            print("\nFeature summary:")
            print(all_features.describe().round(4))
            
            return all_features
            
        except Exception as e:
            print(f"Error in feature processing: {str(e)}")
            raise