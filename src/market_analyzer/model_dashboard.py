"""
Dashboard for visualizing model performance and comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import numpy as np

class ModelPerformanceDashboard:
    """Dashboard for visualizing model performance."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
    
    def plot_ml_metrics(self, results: List[Dict]):
        """Plot ML model metrics comparison."""
        # Extract metrics
        data = []
        for result in results:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                data.append({
                    'model': result['model_name'],
                    'metric': metric,
                    'train': result['train_metrics'][metric],
                    'test': result['test_metrics'][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('ML Models Performance Metrics')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            metric_data = df[df['metric'] == metric]
            
            x = np.arange(len(metric_data['model'].unique()))
            width = 0.35
            
            ax.bar(x - width/2, metric_data['train'],
                  width, label='Train')
            ax.bar(x + width/2, metric_data['test'],
                  width, label='Test')
            
            ax.set_title(metric.title())
            ax.set_xticks(x)
            ax.set_xticklabels(metric_data['model'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_ts_metrics(self, results: List[Dict]):
        """Plot time series model metrics comparison."""
        # Extract metrics
        data = []
        for result in results:
            for metric in ['mse', 'rmse', 'mae', 'mape']:
                data.append({
                    'model': result['model_name'],
                    'metric': metric,
                    'train': result['train_metrics'][metric],
                    'test': result['test_metrics'][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Time Series Models Performance Metrics')
        
        metrics = ['mse', 'rmse', 'mae', 'mape']
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            metric_data = df[df['metric'] == metric]
            
            x = np.arange(len(metric_data['model'].unique()))
            width = 0.35
            
            ax.bar(x - width/2, metric_data['train'],
                  width, label='Train')
            ax.bar(x + width/2, metric_data['test'],
                  width, label='Test')
            
            ax.set_title(metric.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(metric_data['model'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str]):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=self.figsize)
            
            importances = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=True)
            
            importances.plot(kind='barh')
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_learning_curves(self, train_metrics: List[float],
                           test_metrics: List[float],
                           metric_name: str = 'Score'):
        """Plot learning curves."""
        plt.figure(figsize=self.figsize)
        
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'b-', label='Training')
        plt.plot(epochs, test_metrics, 'r-', label='Validation')
        
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, actual: pd.Series,
                                 predictions: pd.Series,
                                 title: str = 'Predictions vs Actual'):
        """Plot model predictions against actual values."""
        plt.figure(figsize=self.figsize)
        
        plt.plot(actual.index, actual.values, 'b-', label='Actual')
        plt.plot(predictions.index, predictions.values, 'r-', label='Predicted')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()