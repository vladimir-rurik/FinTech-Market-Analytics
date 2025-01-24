"""
Core analyzer module for financial market data analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import requests
from .utils import validate_data, detect_outliers

class MarketDataAnalyzer:
    """
    A class for analyzing financial market data including stocks and cryptocurrencies.
    
    Attributes:
        crypto_symbols (list): List of cryptocurrency symbols to analyze
        sp500_symbols (list): List of S&P500 company symbols
        crypto_data (dict): Dictionary storing cryptocurrency historical data
        stock_data (dict): Dictionary storing stock historical data
    """
    
    def __init__(self):
        """Initialize the MarketDataAnalyzer with default cryptocurrency symbols and S&P500 stocks."""
        # List of cryptocurrencies to analyze
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
        # Get S&P500 companies list
        self.sp500_symbols = self._get_sp500_symbols()
        self.crypto_data = {}
        self.stock_data = {}
        
    def _get_sp500_symbols(self) -> List[str]:
        """
        Fetch the list of S&P500 companies from Wikipedia.
        
        Returns:
            List[str]: List of S&P500 company symbols
        """
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except Exception as e:
            print(f"Error fetching S&P500 symbols: {e}")
            return []

    def download_data(self, period: str = "1y") -> None:
        """
        Download historical data for all assets.
        
        Args:
            period (str): Time period to download (e.g., '1y' for one year)
        """
        print("Downloading market data...")
        
        # Download cryptocurrency data
        for symbol in self.crypto_symbols:
            try:
                data = yf.download(symbol, period=period)
                self.crypto_data[symbol] = data
                print(f"Downloaded data for {symbol}")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")

        # Download S&P500 stocks data (using first 10 stocks as example)
        for symbol in self.sp500_symbols[:10]:
            try:
                data = yf.download(symbol, period=period)
                self.stock_data[symbol] = data
                print(f"Downloaded data for {symbol}")
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")

    def check_missing_values(self) -> Dict[str, pd.Series]:
        """
        Check for missing values in the downloaded data.
        
        Returns:
            Dict[str, pd.Series]: Dictionary containing missing value counts for each asset
        """
        missing_data = {}
        
        # Check cryptocurrencies
        for symbol, data in self.crypto_data.items():
            missing = data.isnull().sum()
            if missing.any():
                missing_data[symbol] = missing[missing > 0]
                print(f"\n{symbol} missing values:")
                print(missing[missing > 0])
            else:
                print(f"\n{symbol}: No missing values")

        # Check stocks
        for symbol, data in self.stock_data.items():
            missing = data.isnull().sum()
            if missing.any():
                missing_data[symbol] = missing[missing > 0]
                print(f"\n{symbol} missing values:")
                print(missing[missing > 0])
            else:
                print(f"\n{symbol}: No missing values")
                
        return missing_data

    def detect_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in the price data.
        
        Args:
            method (str): Method for outlier detection ('zscore' or 'iqr')
            threshold (float): Threshold for outlier detection
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing outlier data for each asset
        """
        outliers = {}
        
        # Analyze cryptocurrencies
        for symbol, data in self.crypto_data.items():
            asset_outliers = detect_outliers(data['Close'], method=method, threshold=threshold)
            if asset_outliers.any():
                outliers[symbol] = data[asset_outliers]
                print(f"\n{symbol} potential outliers:")
                print(data[asset_outliers])

        # Analyze stocks
        for symbol, data in self.stock_data.items():
            asset_outliers = detect_outliers(data['Close'], method=method, threshold=threshold)
            if asset_outliers.any():
                outliers[symbol] = data[asset_outliers]
                print(f"\n{symbol} potential outliers:")
                print(data[asset_outliers])
                
        return outliers