"""
Machine learning models for market prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

class MLStrategy:
    """Base class for ML-based trading strategies."""
    
    def __init__(self, model_name: str, model_params: Dict = None):
        """Initialize ML strategy."""
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for ML model.
        
        Args:
            features: DataFrame with features
            target: Series with target values
            
        Returns:
            Tuple of (X, y) arrays ready for ML model
        """
        # Convert categorical columns to numeric
        numeric_features = pd.DataFrame(index=features.index)
        
        for column in features.columns:
            # Skip target columns or date columns
            if column.startswith('future_return_') or isinstance(features[column].dtype, pd.DatetimeTZDtype):
                continue
                
            if features[column].dtype == 'object' or pd.api.types.is_categorical_dtype(features[column]):
                # One-hot encode categorical features
                dummies = pd.get_dummies(features[column], prefix=column)
                numeric_features = pd.concat([numeric_features, dummies], axis=1)
            else:
                # Keep numeric features
                numeric_features[column] = pd.to_numeric(features[column], errors='coerce')
        
        # Store feature columns for later use
        if self.feature_columns is None:
            self.feature_columns = numeric_features.columns.tolist()
        
        # Scale features
        X = self.scaler.fit_transform(numeric_features[self.feature_columns].fillna(0))
        y = target.values
        
        return X, y
    
    def create_target(self, data: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Create target variable based on future returns."""
        # Calculate future returns
        future_returns = data['Close'].pct_change(horizon).shift(-horizon)
        
        # Create target labels (-1: Sell, 0: Hold, 1: Buy)
        target = pd.Series(0, index=data.index)
        target[future_returns > 0.01] = 1  # Buy if >1% return
        target[future_returns < -0.01] = -1  # Sell if <-1% return
        
        return target

class RandomForestStrategy(MLStrategy):
    """Random Forest based trading strategy."""
    
    def __init__(self, model_params: Dict = None):
        """Initialize Random Forest strategy."""
        super().__init__('random_forest', model_params)
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 10,
            'random_state': 42
        }
        self.model_params = {**default_params, **(model_params or {})}
        self.model = RandomForestClassifier(**self.model_params)

class GradientBoostingStrategy(MLStrategy):
    """Gradient Boosting based trading strategy."""
    
    def __init__(self, model_params: Dict = None):
        """Initialize Gradient Boosting strategy."""
        super().__init__('gradient_boosting', model_params)
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        self.model_params = {**default_params, **(model_params or {})}
        self.model = GradientBoostingClassifier(**self.model_params)

class RegularizedLogisticStrategy(MLStrategy):
    """Regularized Logistic Regression based trading strategy."""
    
    def __init__(self, model_params: Dict = None):
        """Initialize Logistic Regression strategy."""
        super().__init__('logistic_regression', model_params)
        default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'multi_class': 'ovr',
            'random_state': 42
        }
        self.model_params = {**default_params, **(model_params or {})}
        self.model = LogisticRegression(**self.model_params)

class MLModelManager:
    """Manager for ML models training and evaluation."""
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize model manager."""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def train_model(self, 
                   strategy: MLStrategy,
                   train_features: pd.DataFrame,
                   train_data: pd.DataFrame,
                   horizon: int = 1) -> Dict:
        """Train ML model and return metrics."""
        # Create target variable
        target = strategy.create_target(train_data, horizon)
        
        # Prepare data
        X, y = strategy.prepare_data(train_features, target)
        
        # Train model
        strategy.model.fit(X, y)
        
        # Save model
        self.save_model(strategy)
        
        # Return training metrics
        y_pred = strategy.model.predict(X)
        return self.calculate_metrics(y, y_pred)
    
    def evaluate_model(self,
                      strategy: MLStrategy,
                      test_features: pd.DataFrame,
                      test_data: pd.DataFrame,
                      horizon: int = 1) -> Dict:
        """Evaluate ML model and return metrics."""
        # Create target variable
        target = strategy.create_target(test_data, horizon)
        
        # Prepare data
        X, y = strategy.prepare_data(test_features, target)
        
        # Make predictions
        y_pred = strategy.model.predict(X)
        
        # Calculate metrics
        return self.calculate_metrics(y, y_pred)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def save_model(self, strategy: MLStrategy):
        """Save trained model to disk."""
        model_path = os.path.join(self.models_dir, f"{strategy.model_name}.joblib")
        joblib.dump({
            'model': strategy.model,
            'scaler': strategy.scaler,
            'feature_columns': strategy.feature_columns
        }, model_path)
    
    def load_model(self, strategy: MLStrategy):
        """Load trained model from disk."""
        model_path = os.path.join(self.models_dir, f"{strategy.model_name}.joblib")
        if os.path.exists(model_path):
            saved_data = joblib.load(model_path)
            strategy.model = saved_data['model']
            strategy.scaler = saved_data['scaler']
            strategy.feature_columns = saved_data['feature_columns']
            return True
        return False