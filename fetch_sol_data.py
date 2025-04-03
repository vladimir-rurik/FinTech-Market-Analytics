from binance_client import fetch_binance_data

def main():
    symbol = "SOLUSDT"
    start = "2020-08-12"
    interval = "1h"

    print(f"Fetching {symbol} data from {start} with interval={interval}...")
    df = fetch_binance_data(symbol, interval=interval, start_date=start)

    # Save to CSV for further analysis
    if df is None or df.empty:
        print("❌ No data fetched or data is empty.")
        return
    print(f"Fetched {len(df)} rows of data.")
    
    df.to_csv("data/SOL_hourly_data.csv", index=False)
    print("✅ Data saved to data/SOL_hourly_data.csv")

if __name__ == "__main__":
    main()
