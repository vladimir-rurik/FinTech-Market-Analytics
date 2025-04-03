"""
Core analyzer module for financial market data analysis, now loading from CSV instead of yfinance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import requests  # might not be strictly needed anymore
import os

class MarketDataAnalyzer:
    """A class for analyzing financial market data loaded from CSV files."""

    def __init__(self):
        """
        Initialize the analyzer with an empty dictionary to store crypto data.
        In the original code, we had sp500, etc., which we've removed.
        """
        self.crypto_data = {}

    def load_csv_data(self, csv_path: str, symbol: str) -> None:
        """
        Load market data from a local CSV file of the form:

        timestamp,open,high,low,close,volume
        2020-08-11 06:00:00,2.85000000,3.47000000,2.85000000,2.95150000,20032.26000000
        ...

        Then store it in self.crypto_data[symbol].
        """
        if not os.path.exists(csv_path):
            print(f"CSV file not found at path: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            # Convert 'timestamp' column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Rename columns to match the code's expected naming
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            # Set index to timestamp
            df.set_index('timestamp', inplace=True)
            # Prepare data
            data = self._prepare_data(df)
            self.crypto_data[symbol] = data
            print(f"Loaded data for {symbol} from {csv_path}")

        except Exception as e:
            print(f"Error loading CSV for {symbol}: {e}")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analysis by handling missing values and ensuring correct types,
        just as in the original code.
        """
        try:
            if df is None or df.empty:
                return pd.DataFrame()

            data = df.copy()

            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, errors='coerce')

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

    def check_data_quality(self) -> Dict:
        """
        Check data quality for all assets (only crypto_data here).
        """
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

        return quality_report

    def get_asset_data(self, symbol: str) -> pd.DataFrame:
        """Get data for a specific asset from self.crypto_data."""
        if symbol in self.crypto_data:
            return self.crypto_data[symbol]
        else:
            raise ValueError(f"Symbol {symbol} not found in crypto_data")
