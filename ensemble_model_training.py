"""
ensemble_model_training.py
Train sub-models (TimeSeriesNN, LLM, RLAgent) for SOL/USDT data,
save the ensemble to 'models/ensemble_strategy.joblib'.
"""

import os
import numpy as np
import pandas as pd
import joblib

from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.experiment_tracker import ExperimentTracker
from market_analyzer.backtester import Backtester

# Sub-strategies for the ensemble
from market_analyzer.ensemble_strategy import (
    TimeSeriesNNStrategy,
    LLMSentimentStrategy,
    RLAgentStrategy,
    EnsembleStrategy
)

def build_lstm_dataset(df: pd.DataFrame, seq_len=30):
    """
    Minimal sample for producing an LSTM dataset from columns 'Open','High','Low','Close','Volume'.
    We create random labels in {0,1,2} => classification placeholder.
    """
    req_cols = ["Open","High","Low","Close","Volume"]
    for c in req_cols:
        if c not in df.columns:
            # If missing, fill with zeros or fetch real data
            df[c] = 0.0

    # Create random labels 0..2
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=len(df))

    # Build rolling windows
    arr = df[req_cols].values
    X_list, y_list = [], []
    for i in range(seq_len, len(df)):
        X_list.append(arr[i-seq_len:i])
        y_list.append(labels[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y

def main():
    data_csv = "data/SOL_hourly_data.csv"
    symbol   = "SOL-USDT"

    if not os.path.exists(data_csv):
        print(f"[Error] {data_csv} not found. Fetch initial data first.")
        return

    # 1) Read CSV, rename columns => uppercase
    df = pd.read_csv(data_csv)
    rename_map = {
        'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'
    }
    df.rename(columns=rename_map, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Initialize analyzer
    analyzer = MarketDataAnalyzer()
    analyzer.crypto_data[symbol] = df
    data = analyzer.get_asset_data(symbol)

    if data.empty or len(data) < 50:
        print("[Error] Not enough data to train. Aborting.")
        return

    # Train/Val/Test split => 60/20/20
    N = len(data)
    train_end = int(N*0.6)
    val_end   = int(N*0.8)
    train_df = data.iloc[:train_end].copy()
    val_df   = data.iloc[train_end:val_end].copy()
    test_df  = data.iloc[val_end:].copy()

    ########## TimeSeriesNNStrategy ##########
    ts_nn = TimeSeriesNNStrategy(name="time_series_nn", seq_length=30, num_features=5, n_hidden=32)
    X_train, y_train = build_lstm_dataset(train_df, seq_len=30)
    X_val,   y_val   = build_lstm_dataset(val_df, seq_len=30)
    if len(X_train) < 1 or len(X_val) < 1:
        print("[TimeSeriesNN] Warning: not enough data for LSTM training. Skipping.")
    else:
        print("[TimeSeriesNN] Training on LSTM dataset...")
        ts_nn.train(X_train, y_train, X_val, y_val, epochs=3, batch_size=16, lr=1e-3)

    ########## LLM Sentiment Strategy ##########
    llm_sent = LLMSentimentStrategy(name="llm_sentiment")
    try:
        llm_sent.load_model()  # requires `pip install transformers`
        print("[LLMSentimentStrategy] Loaded HF sentiment pipeline.")
    except Exception as e:
        print("[LLMSentimentStrategy] Could not load pipeline => will produce zeros. Error:", e)

    ########## RLAgentStrategy ##########
    rl_agent = RLAgentStrategy(name="rl_agent")
    try:
        # Minimal RL training => requires `pip install stable-baselines3`
        rl_agent.train(train_df, timesteps=3000)
        print("[RLAgentStrategy] RL training done.")
    except Exception as e:
        print("[RLAgentStrategy] Could not train => will produce zeros. Error:", e)

    ########## Build Ensemble ##########
    ensemble = EnsembleStrategy(
        sub_strategies=[ts_nn, llm_sent, rl_agent],
        voting=True,
        name="ensemble_strategy"
    )

    # Evaluate on validation set
    backtester = Backtester(data, train_size=0.6, test_size=0.2)
    val_results = backtester.evaluate_strategy(ensemble, backtester.validation_data)
    print("[Ensemble] Validation results:", val_results)

    # Save ensemble to disk
    os.makedirs("models", exist_ok=True)
    model_path = "models/ensemble_strategy.joblib"
    joblib.dump(ensemble, model_path)
    print(f"[Ensemble] Saved to {model_path}")

    # Save experiment metrics if you want
    if isinstance(val_results.get("portfolio_value"), pd.Series):
        val_results["portfolio_value"] = val_results["portfolio_value"].tolist()
    if isinstance(val_results.get("signals"), pd.Series):
        val_results["signals"] = val_results["signals"].tolist()
    if isinstance(val_results.get("returns"), pd.Series):
        val_results["returns"] = val_results["returns"].tolist()

    tracker = ExperimentTracker(results_dir="results")
    tracker.save_experiment(
        experiment_name="ensemble_SOL",
        model_name="EnsembleStrategy",
        train_metrics={},
        test_metrics={},
        validation_metrics=val_results,
        params={"voting": True, "sub_models": ["TimeSeriesNN","LLMSentiment","RLAgent"]}
    )
    print("[Ensemble] Training complete.")

if __name__=="__main__":
    main()
