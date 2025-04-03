"""
hourly_update_and_predict.py
Fetch new SOLUSDT hourly data, append to CSV, rename columns, drop duplicates,
load the trained ensemble and produce signals.
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from binance_client import fetch_binance_data
from market_analyzer.analyzer import MarketDataAnalyzer

def main():
    csv_path = "data/SOL_hourly_data.csv"
    symbol   = "SOLUSDT"
    interval = "1h"
    ensemble_model_path = "models/ensemble_strategy.joblib"

    if not os.path.exists(csv_path):
        print(f"[Error] {csv_path} does not exist. Fetch initial data first.")
        return

    # 1) Find last timestamp => fetch from next hour
    existing_df = pd.read_csv(csv_path)
    if existing_df.empty:
        start_date = "2020-08-12"
    else:
        last_time_str = existing_df["timestamp"].iloc[-1]
        last_time = pd.to_datetime(last_time_str)
        # next hour
        start_time = last_time + pd.Timedelta(hours=1)
        start_date = start_time.strftime("%Y-%m-%d")

    print(f"Fetching new data for {symbol} from {start_date} ...")
    new_data = fetch_binance_data(symbol, interval=interval, start_date=start_date)

    if new_data.empty:
        print("No new data returned from Binance. Exiting.")
        return

    # 2) Append new rows
    new_data.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"✅ Appended {len(new_data)} new rows to {csv_path}")

    # 3) Drop duplicates & rename columns to uppercase
    df = pd.read_csv(csv_path)

    # Drop duplicates by 'timestamp'
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)

    # Rename columns to uppercase
    rename_map = {
        'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'
    }
    df.rename(columns=rename_map, inplace=True)

    df.to_csv(csv_path, index=False)

    # 4) Reload final CSV into a DataFrame
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 5) Load ensemble from disk
    if not os.path.exists(ensemble_model_path):
        print(f"[Error] Model file {ensemble_model_path} not found. Train first.")
        return

    ensemble = joblib.load(ensemble_model_path)
    print("✅ Ensemble strategy loaded from disk.")

    # 6) Pass to analyzer for consistency
    analyzer = MarketDataAnalyzer()
    analyzer.crypto_data["SOL-USDT"] = df
    data = analyzer.get_asset_data("SOL-USDT")

    # 7) Generate signals
    signals = ensemble.generate_signals(data)

    # 8) Print latest signal
    if signals.empty:
        print("[Warning] signals empty, no data?")
        return

    latest_signal = signals.iloc[-1]
    latest_time = signals.index[-1]

    # Map numeric => text
    action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    # if signals can be float, round them:
    discrete_signal = int(round(latest_signal))
    action = action_map.get(discrete_signal, "HOLD")

    print(f"[{latest_time}] New Ensemble Signal = {latest_signal:.2f} => {action}")

if __name__ == "__main__":
    main()
