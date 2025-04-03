# binance_client.py
import requests
import pandas as pd
from datetime import datetime
import time

BASE_URL = 'https://api.binance.com/api/v3/klines'

def fetch_binance_data(symbol, interval='1h', start_date='2018-01-01'):
    """
    Fetch historical OHLCV data from Binance API, renaming columns to match 'Open','High','Low','Close','Volume'.
    """
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp   = int(datetime.now().timestamp() * 1000)
    limit = 1000  # max per request
    all_data = []

    while start_timestamp < end_timestamp:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_timestamp,
            'limit': limit
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print(f"[Error] Binance API returned {response.status_code}")
            break

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        # Next batch => start right after the last candle
        start_timestamp = data[-1][0] + 1
        time.sleep(0.5)  # be nice to API

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Rename columns to the uppercase names your strategies expect
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # Keep only the columns we need
    df = df[['timestamp','Open','High','Low','Close','Volume']]
    return df
