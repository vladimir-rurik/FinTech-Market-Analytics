"""
Trading strategy analysis script (now loads from CSV).
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
    """Optimize strategies using grid search (placeholder). 
       In original code, we might do:
         best_ma_params, _ = backtester.optimize_strategy(...)
       but if that code doesn't exist, remove or adapt as needed.
    """
    # Stub: return some default best params
    return {
        'MA': {'short_window': 20, 'long_window': 50},
        'RSI': {'period': 14, 'oversold': 30, 'overbought': 70},
        'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'BB': {'window': 20, 'num_std': 2.0}
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
    print("Loading market data from CSV...")
    analyzer = MarketDataAnalyzer()
    analyzer.load_csv_data("data/BTC-USD.csv", "BTC-USD")

    for symbol, data in analyzer.crypto_data.items():
        print(f"\nAnalyzing {symbol}...")

        # Initialize backtester
        backtester = Backtester(data)

        # Optimize strategies (placeholder)
        print("Optimizing strategies...")
        best_params = optimize_strategies(backtester)

        # Evaluate strategies on validation set
        print("\nEvaluating strategies on validation set...")
        results = evaluate_strategies(backtester, best_params, backtester.validation_data)

        # Create dashboard and plot results
        print("\nGenerating performance dashboard...")
        dashboard = StrategyDashboard()
        dashboard.plot_portfolio_values(results)
        dashboard.plot_returns_distribution(results)
        dashboard.plot_drawdown(results)
        dashboard.plot_performance_metrics(results)

        print(f"\nAnalysis complete for {symbol}!")

if __name__ == "__main__":
    main()
