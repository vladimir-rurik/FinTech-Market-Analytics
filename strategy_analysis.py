"""
Trading strategy analysis script.
"""

from market_analyzer import MarketDataAnalyzer
from market_analyzer.strategy import (
    MovingAverageCrossStrategy,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy
)
from market_analyzer.backtester import Backtester
from market_analyzer.dashboard import StrategyDashboard
import pandas as pd

def optimize_strategies(backtester):
    """Optimize strategies using grid search."""
    # MA Cross Strategy optimization
    ma_params = {
        'short_window': [10, 20],
        'long_window': [50, 100]
    }
    best_ma_params, _ = backtester.optimize_strategy(
        MovingAverageCrossStrategy,
        ma_params
    )
    
    # RSI Strategy optimization
    rsi_params = {
        'period': [14, 21],
        'oversold': [30],
        'overbought': [70]
    }
    best_rsi_params, _ = backtester.optimize_strategy(
        RSIStrategy,
        rsi_params
    )
    
    # MACD Strategy optimization
    macd_params = {
        'fast_period': [12],
        'slow_period': [26],
        'signal_period': [9]
    }
    best_macd_params, _ = backtester.optimize_strategy(
        MACDStrategy,
        macd_params
    )
    
    # Bollinger Bands Strategy optimization
    bb_params = {
        'window': [20],
        'num_std': [2.0]
    }
    best_bb_params, _ = backtester.optimize_strategy(
        BollingerBandsStrategy,
        bb_params
    )
    
    return {
        'MA': best_ma_params,
        'RSI': best_rsi_params,
        'MACD': best_macd_params,
        'BB': best_bb_params
    }

def evaluate_strategies(backtester, best_params, data):
    """Evaluate strategies with optimized parameters."""
    strategies = [
        MovingAverageCrossStrategy(**best_params['MA']),
        RSIStrategy(**best_params['RSI']),
        MACDStrategy(**best_params['MACD']),
        BollingerBandsStrategy(**best_params['BB'])
    ]
    
    results = []
    for strategy in strategies:
        result = backtester.evaluate_strategy(strategy, data)
        results.append(result)
        
        print(f"\nResults for {strategy.name}:")
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Annual Return: {result['annual_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    
    return results

def main():
    # Initialize analyzer and download data
    print("Downloading market data...")
    analyzer = MarketDataAnalyzer()
    analyzer.download_data(period="2y")  # Use 2 years of data
    
    # Process each cryptocurrency
    for symbol, data in analyzer.crypto_data.items():
        print(f"\nAnalyzing {symbol}...")
        
        # Initialize backtester
        backtester = Backtester(data)
        
        # Optimize strategies
        print("Optimizing strategies...")
        best_params = optimize_strategies(backtester)
        
        # Evaluate strategies on validation set
        print("\nEvaluating strategies on validation set...")
        results = evaluate_strategies(backtester, best_params, 
                                   backtester.validation_data)
        
        # Create dashboard and plot results
        print("\nGenerating performance dashboard...")
        dashboard = StrategyDashboard()
        
        # Generate individual plots
        dashboard.plot_portfolio_values(results)
        dashboard.plot_returns_distribution(results)
        dashboard.plot_drawdown(results)
        dashboard.plot_performance_metrics(results)
        
        # Generate comprehensive summary
        print("\nGenerating strategy summary...")
        dashboard.plot_summary(results)
        
        print(f"\nAnalysis complete for {symbol}!")

if __name__ == "__main__":
    main()