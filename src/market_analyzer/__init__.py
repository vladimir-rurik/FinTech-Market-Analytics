"""
FinTech-Market-Analytics
A Python toolkit for financial market data analysis and preprocessing.
"""

from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.utils import prepare_data, validate_data
from market_analyzer.visualization import plot_market_data, plot_statistics

__version__ = "1.0.0"
__author__ = "Vladimir Rurik"
__email__ = "vladimir.rurik@gmail.com"

__all__ = [
    "MarketDataAnalyzer",
    "prepare_data",
    "validate_data",
    "plot_market_data",
    "plot_statistics",
]