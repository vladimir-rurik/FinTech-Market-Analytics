"""
Data processing and feature engineering pipeline script.
Main script for running data preprocessing, feature engineering, and data quality monitoring.
"""

import os
from market_analyzer import MarketDataAnalyzer
from market_analyzer.preprocessor import DataPreprocessor
from market_analyzer.data_dashboard import DataProcessingDashboard

def main():
    """Run data processing pipeline."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize components
    print("Initializing data processing pipeline...")
    analyzer = MarketDataAnalyzer()
    preprocessor = DataPreprocessor(db_path='data/market_data.db')
    dashboard = DataProcessingDashboard()

    # Download market data
    print("\nDownloading market data...")
    analyzer.download_data(period="2y")

    # Process each asset
    for symbol, data in analyzer.crypto_data.items():
        print(f"\nProcessing {symbol}...")
        
        # Step 1: Data cleaning
        print("Cleaning data...")
        cleaned_data = preprocessor.clean_data(data)
        
        # Step 2: Feature engineering
        print("Generating features...")
        features = preprocessor.engineer_features(cleaned_data)
        print(f"Generated {len(features.columns)} features")
        
        # Step 3: Store processed data
        print("Storing processed data...")
        preprocessor.process_new_data(symbol, cleaned_data)
        
        # Step 4: Generate data quality report
        print("\nGenerating data quality visualizations...")
        dashboard.plot_data_quality(cleaned_data)
        
        # Step 5: Feature analysis
        print("\nAnalyzing features...")
        dashboard.plot_feature_distributions(features)
        dashboard.plot_feature_correlations(features)
        
        # Step 6: Summary dashboard
        print("\nGenerating summary dashboard...")
        dashboard.plot_summary_dashboard(cleaned_data, features)
        
        # Print feature statistics
        print("\nFeature Statistics:")
        print(features.describe().T[['mean', 'std', 'min', 'max']])
        
        # Print data coverage
        print("\nData Coverage:")
        missing_values = cleaned_data.isnull().sum()
        if missing_values.any():
            print("Missing values detected:")
            print(missing_values[missing_values > 0])
        else:
            print("No missing values detected")
            
        print(f"\nProcessing complete for {symbol}!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
    finally:
        print("\nData processing pipeline finished")