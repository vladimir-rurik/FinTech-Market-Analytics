"""
nn_dashboard.py - Example script to display NN strategy results on a dashboard.
"""

from market_analyzer.dashboard import StrategyDashboard
import pandas as pd
import numpy as np

def main():
    # 1) Suppose we have 2 strategy results with all required keys
    # including "annual_return".
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    pv1 = 10000 * pd.Series(np.cumprod(1 + 0.001*np.random.randn(len(dates))), index=dates)
    ret1 = pv1.pct_change().fillna(0)
    results_nn = {
        "strategy_name": "nn_strategy",
        "portfolio_value": pv1,
        "returns": ret1,
        "total_return": float(pv1.iloc[-1]/pv1.iloc[0] - 1),
        "annual_return": 0.12,    # we must provide it
        "volatility": 0.02,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.20
    }

    # Another strategy for comparison
    dates2 = pd.date_range("2025-01-01", periods=50, freq="D")
    pv2 = 10000 * pd.Series(np.cumprod(1 + 0.0005*np.random.randn(len(dates2))), index=dates2)
    ret2 = pv2.pct_change().fillna(0)
    results_macd = {
        "strategy_name": "macd_strategy",
        "portfolio_value": pv2,
        "returns": ret2,
        "total_return": float(pv2.iloc[-1]/pv2.iloc[0] - 1),
        "annual_return": 0.08,
        "volatility": 0.015,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.15
    }

    all_results = [results_nn, results_macd]

    # 2) Plot
    dash = StrategyDashboard()
    dash.plot_portfolio_values(all_results)
    dash.plot_returns_distribution(all_results)
    dash.plot_performance_metrics(all_results)  # no KeyError because we have 'annual_return'
    dash.plot_drawdown(all_results)

if __name__=="__main__":
    main()
