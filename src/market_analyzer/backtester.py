"""
Backtesting module for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .strategy import TradingStrategy

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, data: pd.DataFrame, train_size: float = 0.6, test_size: float = 0.2):
        """Initialize backtester with data splits."""
        # Validate and prepare data
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                raise ValueError("Data index must be convertible to datetime")
        
        self.data = data.copy()
        self.split_data(train_size, test_size)
    
    def split_data(self, train_size: float, test_size: float):
        """Split data into train, test, and validation sets."""
        self.data = self.data.sort_index()  # Ensure data is sorted by date
        total_size = len(self.data)
        train_end = int(total_size * train_size)
        test_end = int(total_size * (train_size + test_size))
        
        self.train_data = self.data.iloc[:train_end].copy()
        self.test_data = self.data.iloc[train_end:test_end].copy()
        self.validation_data = self.data.iloc[test_end:].copy()
    
    def evaluate_strategy(self, 
                         strategy: TradingStrategy, 
                         data: pd.DataFrame,
                         initial_capital: float = 10000.0) -> Dict:
        """Evaluate trading strategy performance."""
        try:
            # Generate signals and calculate returns
            signals = strategy.generate_signals(data)
            if signals is None or signals.empty:
                print(f"Warning: No signals generated for {strategy.name}")
                return self._create_empty_results(strategy.name)
                
            strategy_returns = strategy.calculate_returns(data, signals)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # Calculate portfolio value
            portfolio_value = initial_capital * cumulative_returns
            
            # Calculate metrics
            total_return = (portfolio_value.iloc[-1] - initial_capital) / initial_capital
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            
            daily_returns = portfolio_value.pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            max_drawdown = self.calculate_max_drawdown(portfolio_value)
            
            return {
                'strategy_name': strategy.name,
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'portfolio_value': portfolio_value,
                'signals': signals,
                'returns': strategy_returns
            }
            
        except Exception as e:
            print(f"Error evaluating strategy {strategy.name}: {str(e)}")
            return self._create_empty_results(strategy.name)
    
    def _create_empty_results(self, strategy_name: str) -> Dict:
        """Create empty results for failed strategy evaluation."""
        return {
            'strategy_name': strategy_name,
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'portfolio_value': pd.Series(),
            'signals': pd.Series(),
            'returns': pd.Series()
        }
    
    def calculate_max_drawdown(self, portfolio_value: pd.Series) -> float:
        """Calculate maximum drawdown from portfolio value series."""
        rolling_max = portfolio_value.expanding(min_periods=1).max()
        drawdowns = (portfolio_value - rolling_max) / rolling_max
        return drawdowns.min()
    
    def optimize_strategy(self, 
                         strategy_class,
                         param_grid: Dict,
                         metric: str = 'sharpe_ratio') -> Tuple[Dict, Dict]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_class: Class of the strategy to optimize
            param_grid: Dictionary of parameters to try
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Best parameters and their performance metrics
        """
        best_params = None
        best_metrics = None
        best_score = float('-inf')
        
        # Generate all parameter combinations
        import itertools
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        # Test each combination on the test set
        for params in param_combinations:
            strategy = strategy_class(**params)
            metrics = self.evaluate_strategy(strategy, self.test_data)
            
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics