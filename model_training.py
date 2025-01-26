"""
Model training and evaluation script.
Main script for training and evaluating ML and time series models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from market_analyzer import MarketDataAnalyzer
from market_analyzer.preprocessor import DataPreprocessor
from market_analyzer.ml_models import (
    RandomForestStrategy,
    GradientBoostingStrategy,
    RegularizedLogisticStrategy,
    MLModelManager
)
from market_analyzer.time_series_models import (
    SARIMAXStrategy,
    TimeSeriesModelManager
)
from market_analyzer.data_dashboard import DataProcessingDashboard
from typing import List, Dict

def train_ml_models(features: pd.DataFrame, data: pd.DataFrame,
                   test_features: pd.DataFrame, test_data: pd.DataFrame):
    """Train and evaluate ML models."""
    # Initialize model manager
    model_manager = MLModelManager(models_dir='models/ml')
    
    # Define models to train
    models = [
        RandomForestStrategy(),
        GradientBoostingStrategy(),
        RegularizedLogisticStrategy()
    ]
    
    results = []
    for model in models:
        print(f"\nTraining {model.model_name}...")
        
        # Train model
        train_metrics = model_manager.train_model(
            model, features, data
        )
        print("Training metrics:", train_metrics)
        
        # Evaluate model
        test_metrics = model_manager.evaluate_model(
            model, test_features, test_data
        )
        print("Test metrics:", test_metrics)
        
        results.append({
            'model_name': model.model_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
    
    return results

def train_time_series_models(data: pd.DataFrame, test_data: pd.DataFrame):
    """Train and evaluate time series models."""
    # Initialize model manager
    model_manager = TimeSeriesModelManager(models_dir='models/ts')
    
    # Define models
    models = [
        SARIMAXStrategy()
    ]
    
    results = []
    for model in models:
        print(f"\nTraining {model.model_name}...")
        
        # Train model
        train_metrics = model_manager.train_model(
            model, data
        )
        print("Training metrics:", train_metrics)
        
        # Evaluate model
        test_metrics = model_manager.evaluate_model(
            model, test_data
        )
        print("Test metrics:", test_metrics)
        
        results.append({
            'model_name': model.model_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
    
    return results

def plot_model_comparison(ml_results: List[Dict], ts_results: List[Dict]):
    """Plot model comparison dashboard."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # ML Models Training Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = [r['model_name'] for r in ml_results]
    accuracies = [r['train_metrics']['accuracy'] for r in ml_results]
    ax1.bar(model_names, accuracies)
    ax1.set_title('ML Models - Training Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # ML Models Test Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    test_accuracies = [r['test_metrics']['accuracy'] for r in ml_results]
    ax2.bar(model_names, test_accuracies)
    ax2.set_title('ML Models - Test Accuracy')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Time Series Models Training Metrics
    ax3 = fig.add_subplot(gs[1, 0])
    ts_names = [r['model_name'] for r in ts_results]
    rmse_train = [r['train_metrics']['rmse'] for r in ts_results]
    ax3.bar(ts_names, rmse_train)
    ax3.set_title('Time Series Models - Training RMSE')
    ax3.tick_params(axis='x', rotation=45)
    
    # Time Series Models Test Metrics
    ax4 = fig.add_subplot(gs[1, 1])
    rmse_test = [r['test_metrics']['rmse'] for r in ts_results]
    ax4.bar(ts_names, rmse_test)
    ax4.set_title('Time Series Models - Test RMSE')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run model training and evaluation pipeline."""
    # Create directories
    os.makedirs('models/ml', exist_ok=True)
    os.makedirs('models/ts', exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    analyzer = MarketDataAnalyzer()
    preprocessor = DataPreprocessor(db_path='data/market_data.db')
    
    # Download and process data
    print("\nDownloading market data...")
    analyzer.download_data(period="2y")
    
    ml_results = []
    ts_results = []
    
    # Process each asset
    for symbol, data in analyzer.crypto_data.items():
        print(f"\nProcessing {symbol}...")
        
        # Clean data
        cleaned_data = preprocessor.clean_data(data)
        
        # Generate features
        features = preprocessor.engineer_features(cleaned_data)
        
        # Split data into train and test
        train_size = int(len(cleaned_data) * 0.8)
        train_data = cleaned_data[:train_size]
        test_data = cleaned_data[train_size:]
        train_features = features[:train_size]
        test_features = features[train_size:]
        
        print(f"\nTraining models for {symbol}...")
        
        # Train and evaluate ML models
        ml_res = train_ml_models(train_features, train_data,
                               test_features, test_data)
        ml_results.extend(ml_res)
        
        # Train and evaluate time series models
        ts_res = train_time_series_models(train_data, test_data)
        ts_results.extend(ts_res)
        
        # Plot model comparison
        print("\nGenerating model comparison dashboard...")
        plot_model_comparison(ml_res, ts_res)
        
        print(f"\nProcessing complete for {symbol}!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    finally:
        print("\nModel training pipeline finished")