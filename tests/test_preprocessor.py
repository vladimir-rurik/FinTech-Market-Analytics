"""
Tests for data preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from market_analyzer.preprocessor import DataPreprocessor

def test_handle_missing_values_high_missing():
    """Test handling of data with high percentage of missing values."""
    # Create test data with >30% missing values
    data = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan],  # 50% missing
        'B': [10, 20, np.nan, 40, 50, np.nan, 70, 80, np.nan, 100]     # 30% missing
    })
    
    preprocessor = DataPreprocessor()
    result = preprocessor._handle_missing_values(data)
    
    assert result.isnull().sum().sum() == 0  # No missing values
    assert len(result) == len(data)  # Same length as input
    assert all(col in result.columns for col in data.columns)  # All columns present

def test_handle_missing_values_low_missing():
    """Test handling of data with low percentage of missing values."""
    # Create test data with <30% missing values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],  # 10% missing
        'B': [10, np.nan, 30, 40, 50, 60, 70, 80, 90, 100]  # 10% missing
    })
    
    preprocessor = DataPreprocessor()
    result = preprocessor._handle_missing_values(data)
    
    assert result.isnull().sum().sum() == 0  # No missing values
    assert len(result) == len(data)  # Same length as input
    assert all(col in result.columns for col in data.columns)  # All columns present

def test_handle_missing_values_mixed():
    """Test handling of data with mixed missing value percentages."""
    # Create test data with mixed missing percentages
    data = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan],  # 50% missing
        'B': [10, 20, np.nan, 40, 50, 60, 70, 80, 90, 100]            # 10% missing
    })
    
    preprocessor = DataPreprocessor()
    result = preprocessor._handle_missing_values(data)
    
    assert result.isnull().sum().sum() == 0  # No missing values
    assert len(result) == len(data)  # Same length as input
    assert all(col in result.columns for col in data.columns)  # All columns present

def test_handle_missing_values_no_missing():
    """Test handling of data with no missing values."""
    # Create test data with no missing values
    data = pd.DataFrame({
        'A': range(1, 11),
        'B': range(10, 110, 10)
    })
    
    preprocessor = DataPreprocessor()
    result = preprocessor._handle_missing_values(data)
    
    assert (result == data).all().all()  # Data unchanged
    assert len(result) == len(data)  # Same length as input
    assert all(col in result.columns for col in data.columns)  # All columns present

def test_handle_missing_values_all_missing_column():
    """Test handling of data with a column that's all missing."""
    # Create test data with one column all missing
    data = pd.DataFrame({
        'A': [np.nan] * 10,  # All missing
        'B': range(10, 110, 10)  # No missing
    })
    
    preprocessor = DataPreprocessor()
    result = preprocessor._handle_missing_values(data)
    
    assert result.isnull().sum().sum() == 0  # No missing values
    assert len(result) == len(data)  # Same length as input
    assert all(col in result.columns for col in data.columns)  # All columns present
    assert not (result['A'] == result['A'].iloc[0]).all()  # Not all values same in imputed column

def test_handle_missing_values_time_series():
    """Test handling of time series data with missing values."""
    # Create time series data with missing values
    dates = pd.date_range('2023-01-01', '2023-01-10')
    data = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan],
        'B': [10, 20, np.nan, 40, 50, np.nan, 70, 80, np.nan, 100]
    }, index=dates)
    
    preprocessor = DataPreprocessor()
    result = preprocessor._handle_missing_values(data)
    
    assert result.isnull().sum().sum() == 0  # No missing values
    assert len(result) == len(data)  # Same length as input
    assert all(col in result.columns for col in data.columns)  # All columns present
    assert (result.index == dates).all()  # Time index preserved