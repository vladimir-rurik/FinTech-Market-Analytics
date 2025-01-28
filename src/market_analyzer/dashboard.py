"""
Dashboard for visualizing trading strategy performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

class StrategyDashboard:
    """Dashboard for visualizing and comparing trading strategies."""

    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize

    def plot_portfolio_values(self, strategy_results: List[Dict]):
        """Plot portfolio values for multiple strategies."""
        plt.figure(figsize=self.figsize)
        for result in strategy_results:
            pv = result.get("portfolio_value", pd.Series(dtype=float))
            if not pv.empty:
                plt.plot(pv.index, pv.values, label=result["strategy_name"])
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_returns_distribution(self, strategy_results: List[Dict]):
        """Plot returns distribution for multiple strategies."""
        plt.figure(figsize=self.figsize)
        for result in strategy_results:
            rets = result.get("returns", pd.Series(dtype=float)).dropna()
            if not rets.empty:
                sns.kdeplot(rets, label=result["strategy_name"])
        plt.title("Returns Distribution")
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_performance_metrics(self, strategy_results: List[Dict]):
        """Plot key performance metrics for comparison."""
        # The dashboard expects these keys to exist
        metrics = ["total_return", "annual_return", "volatility", "sharpe_ratio", "max_drawdown"]

        # Build a DataFrame with metrics
        data = []
        for result in strategy_results:
            metrics_dict = {metric: result[metric] for metric in metrics}  # must exist in each result
            metrics_dict["strategy"] = result["strategy_name"]
            data.append(metrics_dict)

        if not data:
            print("No performance metrics to display")
            return

        df_metrics = pd.DataFrame(data)
        # Let's do a simple bar plot for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(data=df_metrics, x="strategy", y=metric, ax=ax)
            ax.set_title(metric.title())
            ax.set_ylabel(metric)
            ax.set_xlabel("")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_drawdown(self, strategy_results: List[Dict]):
        """Plot drawdown over time for multiple strategies."""
        plt.figure(figsize=self.figsize)
        has_data = False
        for result in strategy_results:
            pv = result.get("portfolio_value", pd.Series(dtype=float))
            if not pv.empty:
                rolling_max = pv.cummax()
                drawdown = (pv - rolling_max) / rolling_max
                plt.plot(drawdown.index, drawdown, label=result["strategy_name"])
                has_data = True

        if not has_data:
            print("No drawdown data to display")
            return

        plt.title("Strategy Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
