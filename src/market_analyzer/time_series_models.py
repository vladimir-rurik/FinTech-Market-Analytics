"""
Time series specific models for market prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TimeSeriesStrategy:
    """Base class for time series trading strategies."""
    
    def __init__(self, model_name: str, model_params: Dict = None):
        """Initialize time series strategy."""
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for time series model."""
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Create basic time series features
        features = pd.DataFrame(index=data.index)
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log1p(data['Close']).diff()
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        return features
    
    def create_target(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Create target variable based on future returns."""
        future_returns = data['Close'].pct_change(horizon).shift(-horizon)
        return future_returns

class SARIMAXStrategy(TimeSeriesStrategy):
    """SARIMAX based trading strategy."""
    
    def __init__(self, model_params: Dict = None):
        """Initialize SARIMAX strategy."""
        super().__init__('sarimax', model_params)
        default_params = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 5)
        }
        self.model_params = {**default_params, **(model_params or {})}
    
    def fit(self, data: pd.DataFrame):
        """Fit SARIMAX model."""
        features = self.prepare_data(data)
        self.model = SARIMAX(
            features['returns'],
            exog=features[['volatility']],
            order=self.model_params['order'],
            seasonal_order=self.model_params['seasonal_order']
        ).fit(disp=False)
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Make predictions using SARIMAX model."""
        features = self.prepare_data(data)
        predictions = self.model.forecast(
            steps=horizon,
            exog=features[['volatility']].iloc[-horizon:]
        )
        return predictions

class TimeSeriesModelManager:
    """Manager for time series models training and evaluation."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize model manager."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def train_model(self,
                   strategy: TimeSeriesStrategy,
                   train_data: pd.DataFrame,
                   horizon: int = 1) -> Dict:
        """Train time series model and return metrics."""
        # Fit model
        strategy.fit(train_data)
        
        # Make in-sample predictions
        predictions = strategy.predict(train_data, horizon)
        
        # Calculate target
        target = strategy.create_target(train_data, horizon)
        
        # Calculate metrics
        metrics = self.calculate_metrics(target[:-horizon], predictions)
        
        # Save model
        self.save_model(strategy)
        
        return metrics
    
    def evaluate_model(self,
                      strategy: TimeSeriesStrategy,
                      test_data: pd.DataFrame,
                      horizon: int = 1) -> Dict:
        """Evaluate time series model and return metrics."""
        # Make predictions
        predictions = strategy.predict(test_data, horizon)
        
        # Calculate target
        target = strategy.create_target(test_data, horizon)
        
        # Calculate metrics
        return self.calculate_metrics(target[:-horizon], predictions)
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict:
        """Calculate regression metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def save_model(self, strategy: TimeSeriesStrategy):
        """Save trained model to disk."""
        model_path = os.path.join(self.models_dir, f"{strategy.model_name}.joblib")
        joblib.dump({
            'model': strategy.model,
            'scaler': strategy.scaler,
            'params': strategy.model_params
        }, model_path)
    
    def load_model(self, strategy: TimeSeriesStrategy):
        """Load trained model from disk."""
        model_path = os.path.join(self.models_dir, f"{strategy.model_name}.joblib")
        if os.path.exists(model_path):
            saved_data = joblib.load(model_path)
            strategy.model = saved_data['model']
            strategy.scaler = saved_data['scaler']
            strategy.model_params = saved_data['params']
            return True
        return False