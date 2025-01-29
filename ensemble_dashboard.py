"""
ensemble_dashboard.py - Compare the final ensemble strategy vs. sub-models
in a simple dashboard.
"""

from market_analyzer.dashboard import StrategyDashboard
import pandas as pd

def main():
    # In a real scenario, we'd load sub-model results and ensemble results from
    # either 'experiment_results.json' or re-run the backtester. Here, let's just
    # mock a small dictionary with the required fields.

    # Suppose time_series_nn had total_return=0.08, annual_return=0.1, etc.
    # We'll also create random Series to illustrate.
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    import numpy as np
    import math

    def random_portfolio(dates):
        returns = np.random.normal(0, 0.001, len(dates))
        cum = (1 + returns).cumprod() * 10000
        return pd.Series(cum, index=dates)

    tsnn_pv = random_portfolio(dates)
    llm_pv = random_portfolio(dates)
    rl_pv = random_portfolio(dates)
    ensemble_pv = random_portfolio(dates)

    # Build result dicts with required fields:
    tsnn_results = {
        "strategy_name": "time_series_nn",
        "portfolio_value": tsnn_pv,
        "returns": tsnn_pv.pct_change().fillna(0),
        "total_return": float(tsnn_pv.iloc[-1]/tsnn_pv.iloc[0] - 1),
        "annual_return": 0.12,
        "volatility": 0.02,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.2
    }

    llm_results = {
        "strategy_name": "llm_sentiment",
        "portfolio_value": llm_pv,
        "returns": llm_pv.pct_change().fillna(0),
        "total_return": float(llm_pv.iloc[-1]/llm_pv.iloc[0] - 1),
        "annual_return": 0.10,
        "volatility": 0.015,
        "sharpe_ratio": 1.3,
        "max_drawdown": -0.18
    }

    rl_results = {
        "strategy_name": "rl_agent",
        "portfolio_value": rl_pv,
        "returns": rl_pv.pct_change().fillna(0),
        "total_return": float(rl_pv.iloc[-1]/rl_pv.iloc[0] - 1),
        "annual_return": 0.09,
        "volatility": 0.016,
        "sharpe_ratio": 1.1,
        "max_drawdown": -0.25
    }

    ensemble_results = {
        "strategy_name": "ensemble_strategy",
        "portfolio_value": ensemble_pv,
        "returns": ensemble_pv.pct_change().fillna(0),
        "total_return": float(ensemble_pv.iloc[-1]/ensemble_pv.iloc[0] - 1),
        "annual_return": 0.15,
        "volatility": 0.018,
        "sharpe_ratio": 1.7,
        "max_drawdown": -0.22
    }

    all_results = [tsnn_results, llm_results, rl_results, ensemble_results]

    # 2) Use StrategyDashboard
    from market_analyzer.dashboard import StrategyDashboard
    dash = StrategyDashboard(figsize=(12,10))

    dash.plot_portfolio_values(all_results)
    dash.plot_returns_distribution(all_results)
    dash.plot_performance_metrics(all_results)
    dash.plot_drawdown(all_results)

if __name__=="__main__":
    main()
