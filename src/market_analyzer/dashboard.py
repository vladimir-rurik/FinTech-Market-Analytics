"""
Dashboard for visualizing trading strategy performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple

class StrategyDashboard:
    """Dashboard for visualizing and comparing trading strategies."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
    
    def plot_portfolio_values(self, strategy_results: List[Dict]):
        """Plot portfolio values for multiple strategies."""
        plt.figure(figsize=self.figsize)
        
        for result in strategy_results:
            if not result['portfolio_value'].empty:
                plt.plot(result['portfolio_value'].index,
                        result['portfolio_value'].values,
                        label=result['strategy_name'])
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, strategy_results: List[Dict]):
        """Plot returns distribution for multiple strategies."""
        plt.figure(figsize=self.figsize)
        
        for result in strategy_results:
            returns = result['returns'].dropna()
            if not returns.empty:
                sns.kdeplot(data=returns,
                           label=result['strategy_name'])
        
        plt.title('Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def plot_performance_metrics(self, strategy_results: List[Dict]):
        """Plot key performance metrics for comparison."""
        metrics = ['total_return', 'annual_return', 'volatility', 
                  'sharpe_ratio', 'max_drawdown']
        
        # Create DataFrame with metrics
        data = []
        for result in strategy_results:
            metrics_dict = {metric: result[metric] for metric in metrics}
            metrics_dict['strategy'] = result['strategy_name']
            data.append(metrics_dict)
        
        if not data:
            print("No performance metrics to display")
            return
            
        df_metrics = pd.DataFrame(data)
        
        # Plot metrics
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize)
        fig.suptitle('Strategy Performance Metrics')
        
        for i, metric in enumerate(metrics):
            if df_metrics[metric].notna().any():
                sns.barplot(data=df_metrics, x='strategy', y=metric, ax=axes[i])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, strategy_results: List[Dict]):
        """Plot drawdown over time for multiple strategies."""
        plt.figure(figsize=self.figsize)
        has_data = False
        
        for result in strategy_results:
            if not result['portfolio_value'].empty:
                portfolio_value = result['portfolio_value']
                rolling_max = portfolio_value.expanding(min_periods=1).max()
                drawdown = (portfolio_value - rolling_max) / rolling_max * 100
                
                plt.plot(drawdown.index, drawdown.values,
                        label=f"{result['strategy_name']}")
                has_data = True
        
        if has_data:
            plt.title('Strategy Drawdown Over Time')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.show()
        else:
            print("No drawdown data to display")
    
    def plot_signal_heatmap(self, strategy_results: List[Dict]):
        """Plot signal heatmap for strategies."""
        # Check if there are any valid signals
        valid_signals = {
            result['strategy_name']: result['signals']
            for result in strategy_results
            if not result['signals'].empty
        }
        
        if not valid_signals:
            print("No signals to display in heatmap")
            return
            
        # Combine signals from all strategies
        signals_df = pd.DataFrame(valid_signals)
        
        if signals_df.empty:
            print("No signals to display in heatmap")
            return
            
        plt.figure(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(signals_df.T, cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Signal (-1: Sell, 0: Hold, 1: Buy)'})
        
        plt.title('Trading Signals Comparison')
        plt.xlabel('Date')
        plt.ylabel('Strategy')
        plt.tight_layout()
        plt.show()
    
    def plot_summary(self, strategy_results: List[Dict]):
        """Plot comprehensive summary of all strategies."""
        # Create summary figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # Portfolio Values
        ax1 = fig.add_subplot(gs[0, :])
        for result in strategy_results:
            if not result['portfolio_value'].empty:
                ax1.plot(result['portfolio_value'].index,
                        result['portfolio_value'].values,
                        label=result['strategy_name'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        for result in strategy_results:
            returns = result['returns'].dropna()
            if not returns.empty:
                sns.kdeplot(data=returns,
                           label=result['strategy_name'],
                           ax=ax2)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        # Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        for result in strategy_results:
            if not result['portfolio_value'].empty:
                portfolio_value = result['portfolio_value']
                rolling_max = portfolio_value.expanding(min_periods=1).max()
                drawdown = (portfolio_value - rolling_max) / rolling_max * 100
                ax3.plot(drawdown.index, drawdown.values,
                        label=result['strategy_name'])
        ax3.set_title('Strategy Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Performance Metrics
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        data = []
        for result in strategy_results:
            metrics_dict = {metric: result[metric] for metric in metrics}
            metrics_dict['strategy'] = result['strategy_name']
            data.append(metrics_dict)
            
        if data:
            df_metrics = pd.DataFrame(data)
            ax4 = fig.add_subplot(gs[2, :])
            metric_data = df_metrics.melt(id_vars=['strategy'], 
                                        value_vars=metrics)
            sns.barplot(data=metric_data, x='strategy', y='value', 
                       hue='variable', ax=ax4)
            ax4.set_title('Performance Metrics Comparison')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()