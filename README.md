# FinTech-Market-Analytics

A comprehensive Python toolkit for financial market data analysis and visualization. This project provides tools for automated analysis of stock market and cryptocurrency data, including data downloading, preprocessing, quality analysis, and advanced visualizations.

## Features

- **Automated Data Collection**
  - S&P500 stocks historical data
  - Major cryptocurrencies (BTC, ETH, SOL, XRP) data
  - Automatic handling of missing data and splits

- **Market Analysis Tools**
  - Rolling volatility calculation
  - Drawdown analysis
  - Statistical metrics
  - Returns distribution analysis

- **Advanced Visualizations**
  - Price trend analysis
  - Volatility charts with statistical bands
  - Drawdown analysis with key metrics
  - Correlation matrices
  - Statistical distributions and QQ plots

- **Data Quality Analysis**
  - Missing values detection
  - Outlier analysis and validation
  - Data integrity checks
  - Market anomaly detection

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FinTech-Market-Analytics.git
cd FinTech-Market-Analytics

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows use: venv\Scripts\activate

# Install the package and dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from market_analyzer import MarketDataAnalyzer
from market_analyzer.visualization import (
    plot_market_data,
    plot_statistics,
    plot_correlation_matrix,
    plot_returns_distribution,
    plot_volatility,
    plot_drawdown
)

# Initialize analyzer
analyzer = MarketDataAnalyzer()

# Download and analyze data
analyzer.download_data(period="1y")
analyzer.check_missing_values()
analyzer.detect_outliers()

# Create visualizations
plot_market_data(analyzer.crypto_data, title="Cryptocurrency Price Trends")
plot_volatility(analyzer.crypto_data['BTC-USD'])
plot_drawdown(analyzer.crypto_data['BTC-USD'])
```

## Running the Analysis Script

The project includes a ready-to-use analysis script (`analysis.py`) that demonstrates the main features of the toolkit:

```bash
# Activate virtual environment if you haven't already
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS

# Run the analysis script
python analysis.py
```

The script will:
1. Download market data for cryptocurrencies (BTC, ETH, SOL, XRP) and top S&P500 stocks
2. Perform data quality checks and detect outliers
3. Generate various visualizations:
   - Market price trends
   - Rolling volatility with statistical bands
   - Maximum drawdown analysis
   - Returns distribution with QQ plots
   - Statistical analysis plots
   - Correlation matrices

### Customizing the Analysis

You can modify `analysis.py` to change:
- Time period (default: "1y")
- Assets to analyze
- Window size for volatility calculation
- Types of visualizations to generate

Example of a customized analysis:
```python
from market_analyzer import MarketDataAnalyzer
from market_analyzer.visualization import (
    plot_market_data,
    plot_statistics,
    plot_correlation_matrix,
    plot_returns_distribution,
    plot_volatility,
    plot_drawdown
)

def main():
    # Initialize analyzer
    analyzer = MarketDataAnalyzer()
    
    # Download data with custom period
    analyzer.download_data(period="2y")  # Change period to 2 years
    
    # Check data quality
    analyzer.check_missing_values()
    analyzer.detect_outliers()
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot crypto data
    plot_market_data(
        analyzer.crypto_data,
        title="Cryptocurrency Price Trends",
        figsize=(15, 10)
    )
    
    # Analyze specific cryptocurrency (e.g., Bitcoin)
    if 'BTC-USD' in analyzer.crypto_data:
        btc_data = analyzer.crypto_data['BTC-USD']
        
        # Generate various plots
        plot_statistics(btc_data, figsize=(15, 12))
        plot_correlation_matrix(btc_data)
        plot_returns_distribution(btc_data)
        plot_volatility(btc_data, window=30)  # Change volatility window
        plot_drawdown(btc_data)
    
    print("\nAnalysis complete! Check the generated plots for results.")

if __name__ == "__main__":
    main()
```

### Expected Output

The script will generate multiple plots showing:
1. **Market Overview**:
   - Price trends for all analyzed assets
   - Comparative performance visualization

2. **Volatility Analysis**:
   - Rolling volatility with mean and ±2σ bands
   - Current volatility level
   - Historical volatility trends

3. **Drawdown Analysis**:
   - Historical drawdowns
   - Maximum drawdown points
   - Current drawdown level
   - Mean drawdown reference

4. **Statistical Analysis**:
   - Returns distribution
   - QQ plots for normality testing
   - Key statistical metrics
   - Correlation patterns

5. **Data Quality Indicators**:
   - Missing value reports
   - Outlier detection results
   - Data integrity checks

## Requirements

- Python 3.8+
- Dependencies:
  - yfinance>=0.2.12
  - pandas>=1.5.0
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - statsmodels>=0.13.0
  - scikit-learn>=1.0.0

## Project Structure

```
FinTech-Market-Analytics/
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── market_analyzer/
│       ├── __init__.py
│       ├── analyzer.py      # Core analysis functionality
│       ├── utils.py        # Utility functions
│       └── visualization.py # Plotting functions
└── tests/
│   └── test_analyzer.py
└── analysis.py
```

## Usage Examples

### Basic Market Analysis
```python
analyzer = MarketDataAnalyzer()
analyzer.download_data(period="1y")

# Plot cryptocurrency trends
plot_market_data(analyzer.crypto_data, title="Crypto Market Trends")

# Analyze Bitcoin specifically
btc_data = analyzer.crypto_data['BTC-USD']
plot_statistics(btc_data)
plot_volatility(btc_data)
plot_drawdown(btc_data)
```

### Advanced Analysis
```python
# Get volatility statistics
volatility = analyzer.calculate_rolling_volatility()

# Perform correlation analysis
plot_correlation_matrix(analyzer.crypto_data['BTC-USD'])

# Analyze returns distribution
plot_returns_distribution(btc_data, period='D')
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data provided by Yahoo Finance
- Inspiration from various financial analysis tools and libraries

