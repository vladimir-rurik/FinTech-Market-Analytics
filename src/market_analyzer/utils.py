"""
Utility functions for data preparation and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta

def prepare_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare data for analysis by selecting specific columns and date range.

    Args:
        df: Input DataFrame
        columns: List of columns to keep
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Processed DataFrame
    """
    if columns:
        df = df[columns]
    
    if start_date:
        df = df[df.index >= start_date]
    
    if end_date:
        df = df[df.index <= end_date]
    
    return df

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that `df` is suitable for processing.
    If invalid, raise ValueError.
    Otherwise, return the same DataFrame unmodified.
    """

    if df is None:
        raise ValueError("Received None instead of a DataFrame")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pd.DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    required_cols = ["Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # If all checks pass:
    return df

def calculate_returns(
    df: pd.DataFrame,
    column: str = 'Close',
    period: int = 1
) -> pd.Series:
    """
    Calculate returns for a given period.

    Args:
        df: Input DataFrame
        column: Column to calculate returns for
        period: Period for returns calculation

    Returns:
        Series containing calculated returns
    """
    return df[column].pct_change(period)

def detect_outliers(
    series: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in a data series.

    Args:
        series: Input data series
        method: Method for outlier detection ('zscore' or 'iqr')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    else:
        raise ValueError(f"Unknown method: {method}")

def resample_data(
    df: pd.DataFrame,
    freq: str = 'D',
    agg_dict: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Resample data to a different frequency.

    Args:
        df: Input DataFrame
        freq: Frequency for resampling
        agg_dict: Dictionary of aggregation functions

    Returns:
        Resampled DataFrame
    """
    if agg_dict is None:
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
    
    return df.resample(freq).agg(agg_dict)