"""
Dashboard for neural network trading strategy comparison.
"""

import matplotlib.pyplot as plt
import pandas as pd
from market_analyzer.dashboard import StrategyDashboard

def nn_dashboard_comparison(strategies_results):
    """
    Show performance of multiple strategies, including the neural net one.
    """
    dash = StrategyDashboard()
    dash.plot_portfolio_values(strategies_results)
    dash.plot_returns_distribution(strategies_results)
    dash.plot_performance_metrics(strategies_results)
    dash.plot_drawdown(strategies_results)
    # You can also do more advanced or custom charts
    plt.show()

def main():
    # Suppose we loaded or computed 'nn_results' and 'ma_cross_results', etc.
    strategies_results = [ 
        # each is dict with keys like 'strategy_name','portfolio_value','returns','sharpe_ratio', ...
        # e.g. { "strategy_name": "NeuralNet", "portfolio_value": series, ...}
    ]
    nn_dashboard_comparison(strategies_results)

if __name__ == "__main__":
    main()
