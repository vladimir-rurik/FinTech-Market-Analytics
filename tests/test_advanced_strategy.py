"""
Tests for advanced trading strategy.
"""

import pytest
import pandas as pd
import numpy as np
from market_analyzer.advanced_strategy import AdvancedTechnicalStrategy

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame(index=dates)
    
    # Generate sample price data
    data['Close'] = 100 * (1 + np.random.normal(0, 0.02, len(dates))).cumprod()
    data['Open'] = data['Close'] * (1 + np.random.normal(0, 0.01, len(dates)))
    data['High'] = pd.concat([data['Open'], data['Close']], axis=1).max(axis=1) * \
                   (1 + abs(np.random.normal(0, 0.005, len(dates))))
    data['Low'] = pd.concat([data['Open'], data['Close']], axis=1).min(axis=1) * \
                  (1 - abs(np.random.normal(0, 0.005, len(dates))))
    data['Volume'] = np.random.normal(1000000, 200000, len(dates))
    
    # Add technical indicators
    # RSI
    rsi_period = 14
    price_diff = data['Close'].diff()
    gain = price_diff.where(price_diff > 0, 0).rolling(window=rsi_period).mean()
    loss = -price_diff.where(price_diff < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    data[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    
    # MACD
    data['macd'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    
    # ADX
    data['adx'] = pd.Series(np.random.uniform(0, 100, len(dates)), index=dates)
    
    # Price-volume correlation
    data['price_vol_corr'] = pd.Series(np.random.uniform(-1, 1, len(dates)), index=dates)
    
    # Candlestick patterns
    data['hammer'] = pd.Series(np.random.choice([0, 1], len(dates)), index=dates)
    data['shooting_star'] = pd.Series(np.random.choice([0, 1], len(dates)), index=dates)
    
    # Bollinger Bands
    bb_period = 20
    data[f'bb_width_{bb_period}'] = pd.Series(np.random.uniform(0, 0.1, len(dates)), 
                                             index=dates)
    
    return data

@pytest.fixture
def trend_data():
    """Create sample data with clear trends for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame(index=dates)
    
    # Generate a clear uptrend with small noise
    trend = np.linspace(0, 1, len(dates))  # steadily increasing
    noise = np.random.normal(0, 0.01, len(dates))
    data['Close'] = 100 * (1 + trend + noise)
    data['Open'] = data['Close'] * (1 + np.random.normal(0, 0.005, len(dates)))
    data['High'] = data['Close'] * 1.01
    data['Low']  = data['Close'] * 0.99
    data['Volume'] = 1_000_000 * (1 + trend + np.random.normal(0, 0.1, len(dates)))
    
    # Key indicators used by AdvancedTechnicalStrategy
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['adx'] = 30 + np.random.uniform(0, 20, len(dates))  # always above 25
    data['rsi_14'] = 60 + np.random.uniform(-5, 5, len(dates))  # mostly ~60

    # ADD these to avoid "Error generating signals: 'macd'"
    data['macd'] = 1.5 + np.random.normal(0, 0.05, len(dates))
    data['macd_signal'] = 1.0 + np.random.normal(0, 0.05, len(dates))

    # Ensure volume trend is mostly positive
    data['price_vol_corr'] = np.random.uniform(0.1, 0.9, len(dates))

    # For bullish pattern or squeezes
    data['hammer'] = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
    data['shooting_star'] = np.zeros(len(dates))
    data['bb_width_20'] = np.random.uniform(0.01, 0.05, len(dates))
    
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
    assert all(signals.isin([-1, 0, 1]) | (signals > -2) & (signals < 2))  # Including position sizing

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
    assert all(returns >= -0.02)  # Stop-loss at -2%

def test_position_sizing(sample_data):
    """Test position sizing based on conviction."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(sample_data)
    
    # Check if we have varying position sizes
    unique_signals = signals.unique()
    assert len(unique_signals) > 3  # Should have more than just -1, 0, 1
    assert any(signals > 1) or any(signals < -1)  # Should have increased position sizes

def test_trend_following(trend_data):
    """Test strategy performance in trending market."""
    strategy = AdvancedTechnicalStrategy()
    signals = strategy.generate_signals(trend_data)
    returns = strategy.calculate_returns(trend_data, signals)
    
    # In a strong uptrend, strategy should:
    # 1. Generate more buy signals than sell signals
    buy_signals = (signals > 0).sum()
    sell_signals = (signals < 0).sum()
    assert buy_signals > sell_signals
    
    # 2. Achieve positive returns
    cumulative_return = (1 + returns).prod() - 1
    assert cumulative_return > 0

def test_error_handling_empty_data():
    """Test error handling with empty data."""
    strategy = AdvancedTechnicalStrategy()
    empty_data = pd.DataFrame()
    
    # Should return zero signals for empty data
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
    
    # Calculate basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252/len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Test metrics are within reasonable bounds
    assert -1 < total_return < 10  # Return between -100% and 1000%
    assert -1 < annual_return < 5  # Annual return between -100% and 500%
    assert 0 <= volatility < 1  # Annualized volatility less than 100%
    assert -5 < sharpe_ratio < 5  # Reasonable Sharpe ratio range

def test_strategy_consistency(sample_data):
    """Test strategy consistency with same inputs."""
    strategy = AdvancedTechnicalStrategy()
    
    # Generate signals twice with same data
    signals1 = strategy.generate_signals(sample_data)
    signals2 = strategy.generate_signals(sample_data)
    
    # Signals should be identical
    pd.testing.assert_series_equal(signals1, signals2)

if __name__ == '__main__':
    pytest.main([__file__])