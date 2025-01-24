"""
Visualization functions for market data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
    plt.grid(True)
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
    
    fig, axes = plt.subplots(len(metrics), 2, figsize=figsize)
    fig.suptitle('Statistical Analysis')
    
    for i, metric in enumerate(metrics):
        # Histogram
        sns.histplot(df[metric].dropna(), ax=axes[i, 0])
        axes[i, 0].set_title(f'{metric} Distribution')
        
        # Box plot
        sns.boxplot(y=df[metric].dropna(), ax=axes[i, 1])
        axes[i, 1].set_title(f'{metric} Box Plot')
    
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
    corr = df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
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
    returns = df['Close'].resample(period).last().pct_change().dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Returns distribution
    sns.histplot(returns, kde=True, ax=ax1)
    ax1.set_title(f'{period} Returns Distribution')
    
    # QQ plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    
    plt.tight_layout()
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
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    plt.figure(figsize=figsize)
    plt.plot(volatility.index, volatility, label=f'{window}-day Rolling Volatility')
    plt.title('Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True)
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
    drawdown = (price - peak) / peak
    
    plt.figure(figsize=figsize)
    plt.plot(drawdown.index, drawdown * 100)
    plt.title('Price Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.show()