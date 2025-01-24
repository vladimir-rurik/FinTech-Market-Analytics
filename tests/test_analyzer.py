"""
Tests for the MarketDataAnalyzer class.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.utils import prepare_data, validate_data, calculate_returns, detect_outliers

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = {
        'Open': np.random.normal(100, 10, len(dates)),
        'High': np.random.normal(105, 10, len(dates)),
        'Low': np.random.normal(95, 10, len(dates)),
        'Close': np.random.normal(100, 10, len(dates)),
        'Volume': np.random.normal(1000000, 100000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def analyzer():
    """Create MarketDataAnalyzer instance."""
    return MarketDataAnalyzer()

def test_data_preparation(sample_data):
    """Test data preparation functionality."""
    # Test column selection
    columns = ['Open', 'Close']
    prepared_data = prepare_data(sample_data, columns=columns)
    assert list(prepared_data.columns) == columns
    
    # Test date filtering
    start_date = '2024-06-01'
    end_date = '2024-06-30'
    filtered_data = prepare_data(sample_data, start_date=start_date, end_date=end_date)
    assert filtered_data.index.min().strftime('%Y-%m-%d') == start_date
    assert filtered_data.index.max().strftime('%Y-%m-%d') == end_date

def test_data_validation(sample_data):
    """Test data validation functionality."""
    # Test valid data
    results = validate_data(sample_data)
    assert results['is_valid']
    assert len(results['issues']) == 0
    
    # Test invalid data (with missing values)
    invalid_data = sample_data.copy()
    invalid_data.iloc[0, 0] = np.nan
    results = validate_data(invalid_data)
    assert not results['is_valid']
    assert 'Missing values detected' in results['issues']

def test_returns_calculation(sample_data):
    """Test returns calculation functionality."""
    returns = calculate_returns(sample_data)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_data)
    assert returns.dtype == float

def test_outlier_detection(sample_data):
    """Test outlier detection functionality."""
    # Test z-score method
    outliers_zscore = detect_outliers(sample_data['Close'], method='zscore')
    assert isinstance(outliers_zscore, pd.Series)
    assert outliers_zscore.dtype == bool
    
    # Test IQR method
    outliers_iqr = detect_outliers(sample_data['Close'], method='iqr')
    assert isinstance(outliers_iqr, pd.Series)
    assert outliers_iqr.dtype == bool
    
    # Test invalid method
    with pytest.raises(ValueError):
        detect_outliers(sample_data['Close'], method='invalid_method')

def test_market_data_analyzer(analyzer):
    """Test MarketDataAnalyzer class functionality."""
    # Test initialization
    assert isinstance(analyzer.crypto_symbols, list)
    assert isinstance(analyzer.sp500_symbols, list)
    
    # Test S&P500 symbols retrieval
    assert len(analyzer.sp500_symbols) > 0
    assert all(isinstance(symbol, str) for symbol in analyzer.sp500_symbols)

if __name__ == '__main__':
    pytest.main([__file__])