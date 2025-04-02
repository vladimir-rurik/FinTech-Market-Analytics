import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Binance API Base URL
BASE_URL = 'https://api.binance.com/api/v3/klines'

def fetch_binance_data(symbol, interval='1h', start_date='2018-01-01'):
    """
    Fetch historical hourly OHLCV data from Binance API.
    
    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    :param interval: Timeframe interval ('1h' for hourly data)
    :param start_date: Start date for historical data in 'YYYY-MM-DD' format
    :return: DataFrame containing OHLCV data
    """
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.now().timestamp() * 1000)
    limit = 1000  # Binance max limit per request

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
            print(f"Error fetching data for {symbol}: {response.status_code}")
            break

        data = response.json()
        
        if not data:
            break

        all_data.extend(data)
        start_timestamp = data[-1][0] + 1  # Move to the next batch

        time.sleep(0.5)  # Respect Binance API rate limits

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return df

# List of symbols (Binance format)
symbols = {
    'BTCUSDT': 'BTC',
    'ETHUSDT': 'ETH',
    'ADAUSDT': 'ADA',
    'SOLUSDT': 'SOL',
    'AVAXUSDT': 'AVAX',
    'XLMUSDT': 'XLM',
    'LTCUSDT': 'LTC',
    'DOTUSDT': 'DOT',
    'UNIUSDT': 'UNI',
    'AAVEUSDT': 'AAVE',
    'SANDUSDT': 'SAND',
    'AXSUSDT': 'AXS',
    'MATICUSDT': 'MATIC',
    'FTMUSDT': 'FTM'
}

# Fetch and save data
if __name__ == "__main__":
    for symbol, short_name in symbols.items():
        print(f"Fetching data for {short_name} ({symbol})...")
        df = fetch_binance_data(symbol, start_date='2018-01-01')
        df.to_csv(f'{short_name}_hourly_data.csv', index=False)
        print(f"✅ Data for {short_name} saved successfully!\n")
