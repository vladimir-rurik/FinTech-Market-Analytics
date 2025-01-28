"""
Core analyzer module for financial market data analysis.
"""

import yfinance as yf
import backoff
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import requests

class MarketDataAnalyzer:
    """A class for analyzing financial market data."""
    
    def __init__(self):
        """Initialize the analyzer with cryptocurrency and stock symbols."""
        # List of cryptocurrencies to analyze
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
        # Get S&P500 companies list
        self.sp500_symbols = self._get_sp500_symbols()
        self.crypto_data = {}
        self.stock_data = {}
        
    def _get_sp500_symbols(self) -> List[str]:
        """Fetch the list of S&P500 companies."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except Exception as e:
            print(f"Error fetching S&P500 symbols: {e}")
            return []

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analysis by handling missing values and ensuring correct types.
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            Processed DataFrame
        """
        try:
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Convert price and volume columns to numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Handle missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return pd.DataFrame()

    @staticmethod
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        max_tries=5,
        jitter=None
    )
    def _download_with_backoff(symbol: str, period: str) -> yf.download:
        """
        Download ticker data with exponential backoff on temporary network failures.
        
        Args:
            period: Time period to download (e.g., '1y' for one year)
        """
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    
    def download_data(self, period: str = "1y") -> None:
        print("Downloading market data...")
        
        # Download cryptocurrency data
        for symbol in self.crypto_symbols:
            try:
                data = self._download_with_backoff(symbol, period)
                if not data.empty:
                    self.crypto_data[symbol] = self._prepare_data(data)
                    print(f"Downloaded data for {symbol}")
                else:
                    print(f"No data available for {symbol}")
            
            except Exception as e:
                # If we fail *even after* all backoff retries, log an error once
                print(f"Error downloading {symbol} after retries: {e}")

        # Download S&P500 stocks data (first 10 stocks)
        for symbol in self.sp500_symbols[:10]:
            try:
                data = self._download_with_backoff(symbol, period)
                if not data.empty:
                    self.stock_data[symbol] = self._prepare_data(data)
                    print(f"Downloaded data for {symbol}")
                else:
                    print(f"No data available for {symbol}")
            
            except Exception as e:
                # If we fail *even after* all backoff retries, log an error once
                print(f"Error downloading {symbol} after retries: {e}")

    def check_data_quality(self) -> Dict:
        """Check data quality for all assets."""
        quality_report = {}
        
        def check_asset(data: pd.DataFrame) -> Dict:
            return {
                'missing_values': data.isnull().sum().to_dict() if not data.empty else {},
                'start_date': data.index.min() if not data.empty else None,
                'end_date': data.index.max() if not data.empty else None,
                'total_rows': len(data),
                'unique_dates': data.index.nunique() if not data.empty else 0
            }
        
        # Check crypto data
        for symbol, data in self.crypto_data.items():
            quality_report[symbol] = check_asset(data)
            
        # Check stock data
        for symbol, data in self.stock_data.items():
            quality_report[symbol] = check_asset(data)
            
        return quality_report

    def get_asset_data(self, symbol: str) -> pd.DataFrame:
        """Get data for a specific asset."""
        if symbol in self.crypto_data:
            return self.crypto_data[symbol]
        elif symbol in self.stock_data:
            return self.stock_data[symbol]
        else:
            raise ValueError(f"Symbol {symbol} not found in data")