
import os
import smtplib
from email.mime.text import MIMEText
import pandas as pd
import joblib
from datetime import datetime, timedelta

from binance_client import fetch_binance_data
from market_analyzer.analyzer import MarketDataAnalyzer

# SMTP config via environment variables
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))

# Hardcode your email or set it via env
SMTP_USER = os.environ["SMTP_USER"]
SMTP_PASS = os.environ["SMTP_PASS"]  # forcibly read from environment - no default
RECIPIENT = os.environ["RECIPIENT_EMAIL"]

def send_email(signal_str):
    """
    Sends an email with the given signal info.
    """
    subject = "Crypto Trading Signal"
    body = f"New trading signal: {signal_str}"
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = RECIPIENT

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"[Email] Sent signal notification to {RECIPIENT}")
    except Exception as e:
        print("[Email] Sending failed:", e)

def main():
    csv_path = "data/SOL_hourly_data.csv"
    symbol   = "SOLUSDT"
    interval = "1h"
    ensemble_model_path = "models/ensemble_strategy.joblib"

    # 1) Quick check
    if not os.path.exists(csv_path):
        print(f"[Error] {csv_path} not found; fetch initial data first.")
        return

    # 2) Determine next fetch date/time
    existing_df = pd.read_csv(csv_path)
    if existing_df.empty:
        start_date = "2020-08-12"
    else:
        last_time_str = existing_df["timestamp"].iloc[-1]
        last_time = pd.to_datetime(last_time_str)
        start_date = (last_time + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")

    print(f"Fetching new data for {symbol} from {start_date} ...")
    new_data = fetch_binance_data(symbol, interval=interval, start_date=start_date)
    if new_data.empty:
        print("No new data returned from Binance. Exiting.")
        return

    # 3) Append new data
    new_data.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"✅ Appended {len(new_data)} new rows to {csv_path}")

    # 4) De-duplicate & rename columns
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)

    rename_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    df.rename(columns=rename_map, inplace=True)
    df.to_csv(csv_path, index=False)

    # 5) Reload final CSV
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # 6) Load ensemble
    if not os.path.exists(ensemble_model_path):
        print(f"[Error] {ensemble_model_path} not found. Exiting.")
        return

    ensemble = joblib.load(ensemble_model_path)
    print("✅ Ensemble strategy loaded from disk.")

    # 7) Prepare data
    analyzer = MarketDataAnalyzer()
    analyzer.crypto_data["SOL-USDT"] = df
    data = analyzer.get_asset_data("SOL-USDT")

    # 8) Generate signals
    signals = ensemble.generate_signals(data)
    if signals.empty:
        print("[Warning] signals empty, no data?")
        return

    # 9) Latest signal => Buy/Sell/Hold
    latest_signal = signals.iloc[-1]
    latest_time   = signals.index[-1]
    action_map    = { -1: "SELL", 0: "HOLD", 1: "BUY" }
    discrete_sig  = int(round(latest_signal))
    action        = action_map.get(discrete_sig, "HOLD")

    signal_str = f"[{latest_time}] {action} (raw={latest_signal:.2f})"
    print(signal_str)

    # 10) Send email
    send_email(signal_str)

if __name__ == "__main__":
    main()
