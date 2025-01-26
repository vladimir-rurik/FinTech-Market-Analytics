"""
Dashboard for monitoring data processing and feature engineering.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

class DataProcessingDashboard:
    """Dashboard for monitoring data processing pipeline."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
    
    def plot_data_coverage(self, data: pd.DataFrame):
        """Plot data coverage over time."""
        plt.figure(figsize=self.figsize)
        
        # Plot data points count
        daily_count = data.resample('D').count()['Close']
        plt.plot(daily_count.index, daily_count.values)
        
        plt.title('Data Coverage Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Data Points')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, features: pd.DataFrame):
        """Plot distributions of engineered features."""
        n_features = len(features.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('Feature Distributions')
        
        for i, col in enumerate(features.columns):
            ax = axes[i // n_cols, i % n_cols]
            sns.histplot(features[col].dropna(), ax=ax)
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlations(self, features: pd.DataFrame):
        """Plot correlation matrix of features."""
        plt.figure(figsize=self.figsize)
        
        corr = features.corr()
        mask = np.triu(np.ones_like(corr), k=1)
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5
        )
        
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.show()
    
    def plot_data_quality(self, data: pd.DataFrame):
        """Plot data quality metrics."""
        plt.figure(figsize=self.figsize)
        
        # Calculate daily statistics
        daily_stats = pd.DataFrame({
            'missing': data.isnull().sum(axis=1),
            'volume': data['Volume'],
            'price_range': (data['High'] - data['Low']) / data['Close']
        })
        
        # Plot subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Missing values
        daily_stats['missing'].plot(ax=axes[0])
        axes[0].set_title('Missing Values Over Time')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        daily_stats['volume'].plot(ax=axes[1])
        axes[1].set_title('Trading Volume Over Time')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        # Price range
        daily_stats['price_range'].plot(ax=axes[2])
        axes[2].set_title('Price Range Over Time')
        axes[2].set_ylabel('Range (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_summary_dashboard(self, data: pd.DataFrame, features: pd.DataFrame):
        """Plot comprehensive data processing dashboard."""
        # Create summary figure
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # Data coverage
        ax1 = fig.add_subplot(gs[0, :])
        daily_count = data.resample('D').count()['Close']
        ax1.plot(daily_count.index, daily_count.values)
        ax1.set_title('Data Coverage')
        ax1.set_ylabel('Points per Day')
        ax1.grid(True, alpha=0.3)
        
        # Price and volume
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(data.index, data['Close'])
        ax2.set_title('Price History')
        ax2.set_ylabel('Price')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(data.index, data['Volume'])
        ax3.set_title('Volume History')
        ax3.set_ylabel('Volume')
        ax3.grid(True, alpha=0.3)
        
        # Feature statistics
        ax4 = fig.add_subplot(gs[2, :])
        feature_stats = features.describe().T[['mean', 'std', 'min', 'max']]
        sns.heatmap(feature_stats, annot=True, fmt='.2g', cmap='coolwarm')
        ax4.set_title('Feature Statistics')
        
        plt.tight_layout()
        plt.show()