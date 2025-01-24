"""
Usage of the market_analyzer package.
"""

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
    """Run basic market analysis example."""
    # Initialize analyzer
    analyzer = MarketDataAnalyzer()
    
    # Download data
    print("Downloading market data...")
    analyzer.download_data(period="1y")
    
    # Analyze data quality
    print("\nChecking data quality...")
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
    
    # Plot statistics for Bitcoin
    if 'BTC-USD' in analyzer.crypto_data:
        btc_data = analyzer.crypto_data['BTC-USD']
        plot_statistics(btc_data, figsize=(15, 12))
        plot_correlation_matrix(btc_data)
        plot_returns_distribution(btc_data)
        plot_volatility(btc_data)
        plot_drawdown(btc_data)
    
    print("\nAnalysis complete! Check the generated plots for results.")

if __name__ == "__main__":
    main()