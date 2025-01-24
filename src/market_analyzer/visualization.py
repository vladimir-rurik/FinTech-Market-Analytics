"""
Visualization functions for market data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple

def plot_market_data(
    data: Dict[str, pd.DataFrame],
    title: str = "Market Data Analysis",
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot market data for multiple assets.

    Args:
        data: Dictionary of DataFrames containing market data
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for symbol, df in data.items():
        plt.plot(df.index, df['Close'], label=symbol)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_volatility(
    df: pd.DataFrame,
    window: int = 20,
    figsize: Tuple[int, int] = (15, 7)
) -> None:
    """
    Plot rolling volatility.

    Args:
        df: DataFrame containing market data
        window: Rolling window size
        figsize: Figure size
    """
    returns = df['Close'].pct_change().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    plt.figure(figsize=figsize)
    plt.plot(volatility.index, volatility, label=f'{window}-day Rolling Volatility')
    plt.fill_between(volatility.index, 0, volatility, alpha=0.3)
    plt.title('Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_drawdown(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 7)
) -> None:
    """
    Plot price drawdown.

    Args:
        df: DataFrame containing market data
        figsize: Figure size
    """
    price = df['Close']
    peak = price.expanding(min_periods=1).max()
    drawdown = (price - peak) / peak * 100  # Convert to percentage
    
    plt.figure(figsize=figsize)
    plt.plot(drawdown.index, drawdown)
    plt.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red')
    plt.title('Price Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add min drawdown line and annotation
    min_drawdown = drawdown.min()
    min_drawdown_date = drawdown.idxmin()
    plt.axhline(y=min_drawdown, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Max Drawdown: {min_drawdown:.1f}%', 
                xy=(min_drawdown_date, min_drawdown),
                xytext=(10, 10), textcoords='offset points')
    
    plt.show()

def plot_statistics(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot statistical analysis of market data.

    Args:
        df: DataFrame containing market data
        metrics: List of metrics to plot
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['Close', 'Volume', 'Returns']
    
    # Add returns if not in DataFrame
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()
    
    # Create subplots
    fig, axes = plt.subplots(len(metrics), 2, figsize=figsize)
    if len(metrics) == 1:
        axes = np.array([axes])  # Ensure axes is 2D
    fig.suptitle('Statistical Analysis')
    
    for i, metric in enumerate(metrics):
        data = df[metric].dropna()
        
        # Histogram with KDE
        sns.histplot(data=data, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'{metric} Distribution')
        axes[i, 0].set_xlabel(metric)
        axes[i, 0].set_ylabel('Count')
        
        # Simplified boxplot approach
        axes[i, 1].boxplot(data.values)
        axes[i, 1].set_title(f'{metric} Box Plot')
        axes[i, 1].set_ylabel(metric)
        
        # Add grid
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot correlation matrix for market data.

    Args:
        df: DataFrame containing market data
        figsize: Figure size
    """
    # Calculate correlations for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr), k=1)  # Mask upper triangle
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
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(
    df: pd.DataFrame,
    period: str = 'D',
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot returns distribution and QQ plot.

    Args:
        df: DataFrame containing market data
        period: Returns calculation period ('D' for daily, 'W' for weekly, etc.)
        figsize: Figure size
    """
    # Calculate returns
    returns = df['Close'].pct_change().dropna()
    returns_array = returns.values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Returns distribution
    sns.histplot(data=returns, kde=True, ax=ax1)
    ax1.set_title(f'{period} Returns Distribution')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # QQ plot using statsmodels
    from statsmodels.graphics.gofplots import ProbPlot
    QQ = ProbPlot(returns_array)
    QQ.qqplot(line='45', ax=ax2)
    ax2.set_title('Normal Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_text = (
        f'Mean: {float(returns.mean()):.4f}\n'
        f'Std Dev: {float(returns.std()):.4f}\n'
        f'Skewness: {float(returns.skew()):.4f}\n'
        f'Kurtosis: {float(returns.kurtosis()):.4f}'
    )
    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()