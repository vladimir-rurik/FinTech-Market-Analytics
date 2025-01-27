"""
Model training and evaluation script with visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
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
from market_analyzer.experiment_tracker import ExperimentTracker

def plot_feature_importance(model, feature_names: List[str], title: str):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importance = pd.Series(model.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=True)
        importance.tail(20).plot(kind='barh')  # Show top 20 features
        plt.title(f'Feature Importance - {title}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

def plot_model_comparison(model_results: List[Dict], title: str):
    """Plot model performance comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Model Performance Comparison - {title}')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        data = {
            'Model': [],
            'Train': [],
            'Test': []
        }
        
        for result in model_results:
            data['Model'].append(result['model_name'])
            data['Train'].append(result['train_metrics'][metric])
            data['Test'].append(result['test_metrics'][metric])
        
        df = pd.DataFrame(data)
        df.plot(x='Model', y=['Train', 'Test'], kind='bar', ax=ax)
        ax.set_title(f'{metric.title()} Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(model_result: Dict, data: pd.DataFrame, title: str):
    """Plot model predictions vs actual prices."""
    plt.figure(figsize=(15, 6))
    
    # Plot actual prices
    plt.plot(data.index, data['Close'], label='Actual Price', alpha=0.7)
    
    # Plot buy/sell signals
    signals = model_result['signals']
    buy_signals = signals == 1
    sell_signals = signals == -1
    
    plt.scatter(data.index[buy_signals], data.loc[buy_signals, 'Close'],
                color='green', marker='^', label='Buy Signal')
    plt.scatter(data.index[sell_signals], data.loc[sell_signals, 'Close'],
                color='red', marker='v', label='Sell Signal')
    
    plt.title(f'Trading Signals - {title}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(returns: pd.Series, title: str):
    """Plot distribution of trading returns."""
    plt.figure(figsize=(12, 5))
    
    # Plot histogram with KDE
    sns.histplot(returns, kde=True)
    
    # Add vertical lines for mean and standard deviation
    plt.axvline(returns.mean(), color='r', linestyle='--', 
                label=f'Mean: {returns.mean():.4f}')
    plt.axvline(returns.mean() + returns.std(), color='g', linestyle='--',
                label=f'Mean + Std: {returns.mean() + returns.std():.4f}')
    plt.axvline(returns.mean() - returns.std(), color='g', linestyle='--',
                label=f'Mean - Std: {returns.mean() - returns.std():.4f}')
    
    plt.title(f'Returns Distribution - {title}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model training."""
    try:
        if features.empty:
            raise ValueError("Empty feature set")
            
        # Convert all features to float
        features = features.astype(float)
        
        # Drop any constant columns
        constant_cols = [col for col in features.columns 
                        if features[col].nunique() == 1]
        if constant_cols:
            print(f"Dropping {len(constant_cols)} constant columns")
            features = features.drop(columns=constant_cols)
        
        # Drop columns with too many missing values
        missing_ratio = features.isnull().sum() / len(features)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index
        if len(high_missing_cols) > 0:
            print(f"Dropping {len(high_missing_cols)} columns with >50% missing values")
            features = features.drop(columns=high_missing_cols)
        
        # Fill remaining missing values
        features = features.ffill().bfill().fillna(0)
        
        # Drop highly correlated features
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_cols = [column for column in upper.columns 
                          if any(upper[column] > 0.95)]
            if high_corr_cols:
                print(f"Dropping {len(high_corr_cols)} highly correlated features")
                features = features.drop(columns=high_corr_cols)
        
        print(f"\nFinal feature set shape: {features.shape}")
        print("\nFeature types:")
        print(features.dtypes.value_counts())
        
        # Plot feature correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(features.corr(), cmap='coolwarm', center=0, 
                   annot=False, square=True)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.show()
        
        return features
        
    except Exception as e:
        print(f"Error in feature preparation: {str(e)}")
        raise

def split_data(data: pd.DataFrame, features: pd.DataFrame) -> Tuple:
    """Split data into train, test, and validation sets."""
    # 60% train, 20% test, 20% validation
    train_size = int(len(data) * 0.6)
    test_size = int(len(data) * 0.2)
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]
    
    train_features = features[:train_size]
    test_features = features[train_size:train_size + test_size]
    val_features = features[train_size + test_size:]
    
    return (train_data, test_data, val_data,
            train_features, test_features, val_features)

def main():
    """Run model training and evaluation pipeline."""
    # Create directories
    os.makedirs('models/ml', exist_ok=True)
    os.makedirs('models/ts', exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    analyzer = MarketDataAnalyzer()
    preprocessor = DataPreprocessor(db_path='data/market_data.db')
    experiment_tracker = ExperimentTracker()
    
    # Download and process data
    print("\nDownloading market data...")
    analyzer.download_data(period="2y")
    
    # Process each asset
    for symbol, data in analyzer.crypto_data.items():
        print(f"\nProcessing {symbol}...")
        
        try:
            # Clean data and generate features
            cleaned_data = preprocessor.clean_data(data)
            features = preprocessor.engineer_features(cleaned_data)
            features = prepare_features(features)
            
            # Split data
            splits = split_data(cleaned_data, features)
            train_data, test_data, val_data, train_features, test_features, val_features = splits
            
            print(f"\nTraining models for {symbol}...")
            
            # Train ML models
            ml_model_manager = MLModelManager()
            models = [
                RandomForestStrategy(),
                GradientBoostingStrategy(),
                RegularizedLogisticStrategy()
            ]
            
            ml_results = []
            for model in models:
                print(f"\nTraining {model.model_name}...")
                
                # Train and evaluate
                train_metrics = ml_model_manager.train_model(
                    model, train_features, train_data)
                    
                test_metrics = ml_model_manager.evaluate_model(
                    model, test_features, test_data)
                    
                val_metrics = ml_model_manager.evaluate_model(
                    model, val_features, val_data)
                
                print("Training metrics:", train_metrics)
                print("Test metrics:", test_metrics)
                print("Validation metrics:", val_metrics)
                
                # Save experiment results
                experiment_tracker.save_experiment(
                    experiment_name=symbol,
                    model_name=model.model_name,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    validation_metrics=val_metrics,
                    params=model.model_params
                )
                
                ml_results.append({
                    'model_name': model.model_name,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'validation_metrics': val_metrics,
                    'model': model
                })
            
            # Plot experiment history
            experiment_tracker.plot_experiment_history(symbol)
            
            # Plot model comparison
            plot_model_comparison(ml_results, symbol)
            
            # Get and plot best model
            best_model = experiment_tracker.get_best_experiment(symbol)
            print(f"\nBest model for {symbol}:", 
                  best_model['model_name'])
            print("Best validation metrics:", 
                  best_model['validation_metrics'])
            
            print(f"\nProcessing complete for {symbol}!")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    finally:
        print("\nModel training pipeline finished")
    
    # Process each asset
    for symbol, data in analyzer.crypto_data.items():
        print(f"\nProcessing {symbol}...")
        
        try:
            # Clean data
            cleaned_data = preprocessor.clean_data(data)
            
            # Generate features
            features = preprocessor.engineer_features(cleaned_data)
            
            # Plot raw data
            plt.figure(figsize=(15, 6))
            plt.plot(cleaned_data.index, cleaned_data['Close'])
            plt.title(f'{symbol} Price History')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Prepare features for modeling
            features = prepare_features(features)
            
            # Split data into train and test
            train_size = int(len(cleaned_data) * 0.8)
            train_data = cleaned_data[:train_size]
            test_data = cleaned_data[train_size:]
            train_features = features[:train_size]
            test_features = features[train_size:]
            
            print(f"\nTraining models for {symbol}...")
            
            # Train ML models
            ml_model_manager = MLModelManager()
            models = [
                RandomForestStrategy(),
                GradientBoostingStrategy(),
                RegularizedLogisticStrategy()
            ]
            
            ml_results = []
            for model in models:
                print(f"\nTraining {model.model_name}...")
                
                # Train and evaluate
                train_metrics = ml_model_manager.train_model(
                    model, train_features, train_data)
                test_metrics = ml_model_manager.evaluate_model(
                    model, test_features, test_data)
                
                print("Training metrics:", train_metrics)
                print("Test metrics:", test_metrics)
                
                ml_results.append({
                    'model_name': model.model_name,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'model': model
                })
                
                # Plot feature importance for tree-based models
                if hasattr(model.model, 'feature_importances_'):
                    plot_feature_importance(model.model, 
                                         features.columns, 
                                         f"{symbol} - {model.model_name}")
            
            # Plot model comparison
            plot_model_comparison(ml_results, symbol)
            
            # Plot predictions for best model
            best_model = max(ml_results, 
                           key=lambda x: x['test_metrics']['f1'])
            plot_predictions(best_model, test_data,
                           f"{symbol} - {best_model['model_name']}")
            
            print(f"\nProcessing complete for {symbol}!")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
    finally:
        print("\nModel training pipeline finished")