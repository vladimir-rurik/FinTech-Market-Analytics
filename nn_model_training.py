"""
Script: Train and test the NeuralNetworkStrategy.
"""

import pandas as pd
import numpy as np

from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.nn_strategy import NeuralNetworkStrategy
from market_analyzer.backtester import Backtester
from market_analyzer.experiment_tracker import ExperimentTracker

def main():
    # 1. Load data
    analyzer = MarketDataAnalyzer()
    analyzer.download_data(period="1y")
    data = analyzer.get_asset_data("BTC-USD")

    # 2. Create 'label' col for classification (0/1/2)
    data["future_return"] = data["Close"].shift(-1) / data["Close"] - 1
    data["label"] = 1  # default = Hold
    data.loc[data["future_return"] > 0.01, "label"] = 2  # Buy
    data.loc[data["future_return"] < -0.01, "label"] = 0  # Sell
    data.dropna(subset=["future_return"], inplace=True)
    data.drop(columns=["future_return"], inplace=True)

    # 3. Split data
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_df = data.iloc[:train_end].copy()
    val_df   = data.iloc[train_end:val_end].copy()
    test_df  = data.iloc[val_end:].copy()

    # 4. Prepare LSTM data
    X_train, y_train = prepare_lstm_data(train_df, seq_length=30)
    X_val, y_val     = prepare_lstm_data(val_df, seq_length=30)

    # 5. Create and train strategy
    params = {
        "seq_length": 30,
        "num_features": 5,  # because we might use 5 columns in prepare_lstm_data
        "n_hidden": 64,
        "epochs": 5,
        "batch_size": 16
    }
    nn_strat = NeuralNetworkStrategy(params=params)
    metrics = nn_strat.train(X_train, y_train, X_val, y_val)
    print("NN training done. Val metrics:", metrics)

    # 6. Evaluate on validation set with backtester
    backtester = Backtester(val_df, train_size=1.0, test_size=0.0)
    val_results = backtester.evaluate_strategy(nn_strat, backtester.train_data)
    print("Val results:", val_results)

    # 7. Save or convert any Series in val_results to lists so we can store them in JSON
    for key in ["signals", "portfolio_value", "returns"]:
        if isinstance(val_results.get(key), pd.Series):
            val_results[key] = val_results[key].tolist()

    # 8. Log with experiment tracker
    tracker = ExperimentTracker(results_dir="results")
    tracker.save_experiment(
        experiment_name="nn_trading",
        model_name="NeuralNetworkStrategy",
        train_metrics={"val_acc": metrics["final_val_acc"]},
        test_metrics={"sharpe": val_results["sharpe_ratio"]},
        validation_metrics=val_results,
        params=params
    )

def prepare_lstm_data(df: pd.DataFrame, seq_length: int = 30):
    """
    Create X,y for LSTM classification in {0,1,2}.
    We assume df has ["Close","High","Low","Open","Volume","label"].
    We'll use the first 5 as features, label as 0/1/2 => one-hot.
    """
    feature_cols = ["Close","High","Low","Open","Volume"]
    if any(col not in df.columns for col in feature_cols + ["label"]):
        return None, None

    data_values = df[feature_cols].values  # shape (N,5)
    labels = df["label"].values  # shape (N,)

    X, y = [], []
    for i in range(seq_length, len(df)):
        # window of shape (seq_length, 5)
        window = data_values[i-seq_length : i]
        X.append(window)
        label_idx = labels[i]
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[label_idx] = 1.0
        y.append(one_hot)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

if __name__ == "__main__":
    main()
