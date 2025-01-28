"""
Tests for advanced trading strategy using TA-Lib.
"""

import pytest
import pandas as pd
import numpy as np
from market_analyzer.advanced_strategy import AdvancedTechnicalStrategy

@pytest.fixture
def sample_data():
    """
    Create sample data with a forced CRASH in first 30 days and
    a forced SURGE in next 10 days, so RSI/MACD produce buy/sell signals
    with scaled convictions. 
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    data = pd.DataFrame(index=dates)
    
    n = len(dates)
    close = np.zeros(n)
    close[0] = 100.0

    # 1) CRASH from day1..29 => each day -3%
    for i in range(1, 30):
        close[i] = close[i - 1] * 0.97  # big downward

    # 2) SURGE from day30..39 => each day +4%
    close[30] = close[29]
    for i in range(31, 40):
        close[i] = close[i - 1] * 1.04

    # 3) The rest day40..end => mild random (±0.2% daily)
    for i in range(40, n):
        pct = np.random.normal(0, 0.002)  # small daily drift
        close[i] = close[i - 1] * (1 + pct)

    data["Close"] = close

    # Make 'High' a few % above 'Close', 'Low' a few % below
    data["High"] = data["Close"] * (1 + np.random.uniform(0.005, 0.01, n))
    data["Low"]  = data["Close"] * (1 - np.random.uniform(0.005, 0.01, n))
    data["Open"] = (data["High"] + data["Low"]) / 2.0
    data["Volume"] = np.random.normal(1_000_000, 200_000, n)

    return data

@pytest.fixture
def trend_data():
    """
    Create a strong uptrend overall, with a small mid dip, ensuring
    more total buy signals than sells.
    """
    np.random.seed(43)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    data = pd.DataFrame(index=dates)
    
    n = len(dates)
    close = np.zeros(n)
    close[0] = 100.0

    # 1) Up from day1..59 => +0.6% daily
    for i in range(1, 60):
        close[i] = close[i - 1] * 1.006

    # 2) Small dip day60..70 => -0.4% daily
    for i in range(60, 70):
        close[i] = close[i - 1] * 0.996

    # 3) Then up again day70..end => +0.5% daily
    for i in range(70, n):
        close[i] = close[i - 1] * 1.005

    data["Close"] = close
    data["High"] = data["Close"] * (1 + np.random.uniform(0.005, 0.01, n))
    data["Low"]  = data["Close"] * (1 - np.random.uniform(0.005, 0.01, n))
    data["Open"] = (data["High"] + data["Low"]) / 2.0
    data["Volume"] = np.random.normal(1_000_000, 200_000, n)

    return data

def test_strategy_initialization():
    """Test strategy initialization with default parameters."""
    strategy = AdvancedTechnicalStrategy()
    assert strategy.model_name == 'advanced_technical'
    assert 'rsi_period' in strategy.params
    assert 'adx_threshold' in strategy.params

def test_strategy_initialization_with_params():
    """Test strategy initialization with custom parameters."""
    params = {'rsi_period': 21, 'adx_threshold': 30}
    strategy = AdvancedTechnicalStrategy(params)
    assert strategy.params['rsi_period'] == 21
    assert strategy.params['adx_threshold'] == 30

def test_signal_generation(sample_data):
    """Test signal generation."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(sample_data)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(sample_data)
    assert signals.index.equals(sample_data.index)
    # Must be in [-2,2] or so, or we allow the test to pass if it is ±(1.x)
    assert (signals >= -2).all() and (signals <= 2).all(), "Signals out of expected range"

def test_returns_calculation(sample_data):
    """Test returns calculation."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(sample_data)
    returns = strategy.calculate_returns(sample_data, signals)
    
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_data)
    assert returns.index.equals(sample_data.index)
    assert not returns.isnull().any()

def test_stop_loss(sample_data):
    """Test stop-loss implementation."""
    strategy = AdvancedTechnicalStrategy()
    signals = pd.Series(1, index=sample_data.index)  # Always long
    returns = strategy.calculate_returns(sample_data, signals)
    
    # Check if returns are capped at stop-loss level
    assert (returns >= -0.02).all(), "Stop-loss not enforced at -2%"

def test_position_sizing(sample_data):
    """Test position sizing based on conviction."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(sample_data)
    
    unique_signals = signals.unique()
    # Must have more than just -1, 0, 1
    assert len(unique_signals) > 3, f"No extra sized signals found. unique={unique_signals}"
    # Must have signals bigger than 1 or less than -1
    assert any(signals > 1) or any(signals < -1), "No scaled signals beyond ±1 found."

def test_trend_following(trend_data):
    """Test strategy performance in trending market."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(trend_data)
    returns = strategy.calculate_returns(trend_data, signals)
    
    # 1) More buys than sells
    buy_signals = (signals > 0).sum()
    sell_signals = (signals < 0).sum()
    assert buy_signals > sell_signals, f"More sells than buys in an uptrend! buy={buy_signals}, sell={sell_signals}"
    
    # 2) Positive return
    cumret = (1 + returns).prod() - 1
    assert cumret > 0, f"Uptrend strategy produced negative return! {cumret}"

def test_error_handling_empty_data():
    """Test error handling with empty data."""
    strategy = AdvancedTechnicalStrategy()
    empty_data = pd.DataFrame()
    
    signals = strategy.generate_signals(empty_data)
    assert len(signals) == 0

def test_error_handling_missing_columns():
    """Test error handling with missing columns."""
    strategy = AdvancedTechnicalStrategy()
    invalid_data = pd.DataFrame({'Close': [100, 101, 102]})
    signals = strategy.generate_signals(invalid_data)
    assert all(signals == 0)

def test_error_handling_invalid_data():
    """Test error handling with invalid data types."""
    strategy = AdvancedTechnicalStrategy()
    invalid_data = pd.DataFrame({
        'Close': ['invalid', 'data', 'type'],
        'Volume': [1, 2, 3]
    })
    signals = strategy.generate_signals(invalid_data)
    assert all(signals == 0)

def test_strategy_performance_metrics(sample_data):
    """Test strategy performance metrics calculation."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(sample_data)
    returns = strategy.calculate_returns(sample_data, signals)
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / vol if vol != 0 else 0
    
    assert -1 < total_return < 10, f"Total return out of range: {total_return}"
    assert -1 < annual_return < 5, f"Annual return out of range: {annual_return}"
    assert 0 <= vol < 1, f"Vol out of range: {vol}"
    assert -5 < sharpe < 5, f"Sharpe out of range: {sharpe}"

def test_strategy_consistency(sample_data):
    """Test strategy consistency with same inputs."""
    strategy = AdvancedTechnicalStrategy()
    s1 = strategy.generate_signals(sample_data)
    s2 = strategy.generate_signals(sample_data)
    pd.testing.assert_series_equal(s1, s2)

if __name__ == "__main__":
    pytest.main([__file__])
