"""
Advanced trading strategy analysis script.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from market_analyzer import MarketDataAnalyzer
from market_analyzer.preprocessor import DataPreprocessor
from market_analyzer.experiment_tracker import ExperimentTracker
from market_analyzer.advanced_strategy import AdvancedTechnicalStrategy
from market_analyzer.ml_models import MLModelManager

def plot_strategy_performance(data: pd.DataFrame, signals: pd.Series, returns: pd.Series, title: str):
    """Plot strategy performance with entry/exit points."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot price and signals
    ax1.plot(data.index, data['Close'], label='Price', alpha=0.7)
    
    # Plot buy signals
    buy_signals = signals > 0
    sell_signals = signals < 0
    
    ax1.scatter(data.index[buy_signals], data.loc[buy_signals, 'Close'],
                marker='^', color='green', label='Buy', alpha=0.7)
    ax1.scatter(data.index[sell_signals], data.loc[sell_signals, 'Close'],
                marker='v', color='red', label='Sell', alpha=0.7)
    
    ax1.set_title(f'Trading Signals - {title}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    ax2.plot(cumulative_returns.index, cumulative_returns, label='Strategy Returns')
    ax2.set_title('Cumulative Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_risk_metrics(returns: pd.Series, title: str):
    """Plot risk metrics and distributions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Returns distribution
    returns.hist(bins=50, ax=ax1)
    ax1.set_title('Returns Distribution')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    
    # Rolling volatility
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
    rolling_vol.plot(ax=ax2)
    ax2.set_title('Rolling Volatility (20-day)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility')
    
    # Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    drawdown.plot(ax=ax3)
    ax3.set_title('Drawdown')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    
    # Rolling Sharpe ratio
    risk_free_rate = 0.02  # 2% annual risk-free rate
    excess_returns = returns - risk_free_rate/252
    rolling_sharpe = (excess_returns.rolling(window=252).mean() * 252) / \
                    (excess_returns.rolling(window=252).std() * np.sqrt(252))
    rolling_sharpe.plot(ax=ax4)
    ax4.set_title('Rolling Sharpe Ratio (1-year)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sharpe Ratio')
    
    plt.suptitle(f'Risk Metrics - {title}')
    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(returns: pd.Series) -> dict:
    """Calculate comprehensive performance metrics."""
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252/len(returns)) - 1
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Win rate analysis
    winning_days = returns > 0
    win_rate = winning_days.mean()
    
    # Risk/return ratios
    sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(252)) \
                    if len(returns[returns < 0]) > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }

def main():
    """Run advanced strategy analysis."""
    # Create directories
    os.makedirs('results', exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    analyzer = MarketDataAnalyzer()
    preprocessor = DataPreprocessor()
    experiment_tracker = ExperimentTracker()
    
    # Download market data
    print("\nDownloading market data...")
    analyzer.download_data(period="2y")
    
    # Define strategy parameters to test
    strategy_params = [
        {'rsi_period': 14, 'adx_threshold': 25},
        {'rsi_period': 21, 'adx_threshold': 20},
        {'rsi_period': 9, 'adx_threshold': 30}
    ]
    
    # Process each asset
    for symbol, data in analyzer.crypto_data.items():
        print(f"\nAnalyzing {symbol}...")
        
        try:
            # Clean and prepare data
            cleaned_data = preprocessor.clean_data(data)
            features = preprocessor.engineer_features(cleaned_data)
            
            # Test different strategy parameters
            best_metrics = None
            best_strategy = None
            
            for params in strategy_params:
                print(f"\nTesting parameters: {params}")
                strategy = AdvancedTechnicalStrategy(params)
                
                # Generate signals
                signals = strategy.generate_signals(features)
                returns = strategy.calculate_returns(cleaned_data, signals)
                
                # Calculate metrics
                metrics = calculate_performance_metrics(returns)
                print("\nPerformance Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                # Track experiment
                experiment_tracker.save_experiment(
                    experiment_name=f"{symbol}_advanced",
                    model_name=f"advanced_technical_{params}",
                    train_metrics=metrics,
                    test_metrics=metrics,  # In this case, we're not doing train/test split
                    validation_metrics=metrics,
                    params=params
                )
                
                # Update best strategy
                if (best_metrics is None or 
                    metrics['sharpe_ratio'] > best_metrics['sharpe_ratio']):
                    best_metrics = metrics
                    best_strategy = strategy
                    best_signals = signals
                    best_returns = returns
            
            # Plot results for best strategy
            print("\nPlotting results for best strategy...")
            plot_strategy_performance(cleaned_data, best_signals, best_returns, symbol)
            plot_risk_metrics(best_returns, symbol)
            
            # Save results
            results = pd.DataFrame({
                'Date': cleaned_data.index,
                'Price': cleaned_data['Close'],
                'Signal': best_signals,
                'Returns': best_returns
            })
            results.to_csv(f'results/{symbol}_advanced_strategy.csv')
            
            print(f"\nAnalysis complete for {symbol}!")
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
    finally:
        print("\nAdvanced strategy analysis complete")