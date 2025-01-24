"""
Visualization functions for market data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from statsmodels.graphics.gofplots import ProbPlot

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
    plt.plot(volatility.index, volatility, 
             color='blue', label=f'{window}-day Rolling Volatility')
    
    # Add mean and standard deviation lines
    mean_vol = float(volatility.mean())
    std_vol = float(volatility.std())
    plt.axhline(y=mean_vol, color='red', linestyle='--', alpha=0.5, 
                label=f'Mean: {mean_vol:.3f}')
    plt.axhline(y=mean_vol + 2*std_vol, color='gray', linestyle=':', alpha=0.5,
                label='+2σ')
    plt.axhline(y=max(0, mean_vol - 2*std_vol), color='gray', linestyle=':', alpha=0.5,
                label='-2σ')
    
    plt.title('Rolling Volatility Analysis')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add current volatility annotation
    current_vol = float(volatility.iloc[-1])
    plt.annotate(f'Current: {current_vol:.3f}', 
                xy=(volatility.index[-1], current_vol),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
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
    returns = df['Close'].resample(period).last().pct_change().dropna()
    returns_array = returns.values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Returns distribution
    sns.histplot(data=returns, kde=True, ax=ax1)
    ax1.set_title(f'{period} Returns Distribution')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # QQ plot
    QQ = ProbPlot(returns_array)
    QQ.qqplot(line='45', ax=ax2)
    ax2.set_title('Normal Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # Calculate statistics
    mean_val = returns.mean().item()
    std_val = returns.std().item()
    skew_val = returns.skew().item()
    kurt_val = returns.kurtosis().item()
    
    # Add statistics annotation
    stats_text = (
        f'Mean: {mean_val:.4f}\n'
        f'Std Dev: {std_val:.4f}\n'
        f'Skewness: {skew_val:.4f}\n'
        f'Kurtosis: {kurt_val:.4f}'
    )
    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
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
    plt.plot(drawdown.index, drawdown, color='red', label='Drawdown')
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add statistics
    min_drawdown = float(drawdown.min())
    min_drawdown_date = drawdown.idxmin()
    mean_drawdown = float(drawdown.mean())
    current_drawdown = float(drawdown.iloc[-1])
    
    # Add min drawdown line and annotation
    plt.axhline(y=min_drawdown, color='red', linestyle='--', alpha=0.5,
                label=f'Max Drawdown: {min_drawdown:.1f}%')
    plt.axhline(y=mean_drawdown, color='gray', linestyle=':', alpha=0.5,
                label=f'Mean Drawdown: {mean_drawdown:.1f}%')
    
    # Add current drawdown annotation
    plt.annotate(f'Current: {current_drawdown:.1f}%', 
                xy=(drawdown.index[-1], current_drawdown),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.title('Price Drawdown Analysis')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show()