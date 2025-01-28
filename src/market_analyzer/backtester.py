"""
Backtesting module for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict
from .strategy import TradingStrategy

class Backtester:
    """Backtesting engine for trading strategies."""

    def __init__(self, data: pd.DataFrame, train_size: float = 0.6, test_size: float = 0.2):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame")

        self.data = data.copy().sort_index()
        self.split_data(train_size, test_size)

    def split_data(self, train_size: float, test_size: float):
        n = len(self.data)
        train_end = int(n * train_size)
        test_end  = int(n * (train_size + test_size))

        self.train_data = self.data.iloc[:train_end].copy()
        self.test_data  = self.data.iloc[train_end:test_end].copy()
        self.validation_data = self.data.iloc[test_end:].copy()

    def evaluate_strategy(self, strategy: TradingStrategy, data: pd.DataFrame) -> Dict:
        """
        Evaluate the strategy on 'data'. Return a dictionary with all needed fields,
        including 'annual_return', so the dashboard won't fail.
        """
        results = self._create_empty_results(strategy.name)
        try:
            signals = strategy.generate_signals(data)
            print("DEBUG signals type:", type(signals))  # should be <class 'pandas.core.series.Series'>

            if signals.empty:
                print(f"No signals generated for {strategy.name}")
                return results

            # Calculate returns from strategy
            strategy_returns = strategy.calculate_returns(data, signals)

            # Build portfolio value
            initial_capital = 10000.0
            cum_returns = (1 + strategy_returns).cumprod()
            portfolio_value = initial_capital * cum_returns

            # Compute metrics
            total_return = (portfolio_value.iloc[-1] - initial_capital)/initial_capital

            daily_ret = portfolio_value.pct_change().dropna()
            if len(daily_ret) < 1:
                return results

            volatility = daily_ret.std() * np.sqrt(252)
            # annual_return
            annual_return = (1 + total_return)**(252/len(daily_ret)) - 1
            sharpe_ratio = annual_return / volatility if volatility!=0 else 0.0
            max_dd = self._calculate_max_drawdown(portfolio_value)

            results.update({
                "total_return": float(total_return),
                "annual_return": float(annual_return),  # ensures we have 'annual_return'
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_dd),
                "portfolio_value": portfolio_value,
                "signals": signals,
                "returns": strategy_returns
            })

        except Exception as e:
            print(f"Error evaluating strategy {strategy.name}: {e}")

        return results

    def _create_empty_results(self, strategy_name: str) -> Dict:
        return {
            "strategy_name": strategy_name,
            "total_return": 0.0,
            "annual_return": 0.0,     # also included by default
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "portfolio_value": pd.Series(dtype=float),
            "signals": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float)
        }

    def _calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        roll_max = portfolio_value.cummax()
        drawdown = (portfolio_value - roll_max) / roll_max
        return float(drawdown.min())
