from okx_api import Market
import pandas as pd
import time

def date_to_timestamp(date_str):
    """Convert a date string to a timestamp."""
    return int(pd.Timestamp(date_str).timestamp() * 1000)

def fetch_data_between_dates(instId, start_date, end_date, bar, limit=100):
    """Fetch historical data between start_date and end_date."""
    
    market = Market()
    all_data = []
    
    # Convert start_date and end_date to timestamps
    start_timestamp = date_to_timestamp(start_date)
    end_timestamp = date_to_timestamp(end_date)
    
    bar_to_milliseconds = {
        '1m': 60000,
        '5m': 300000,
        '15m': 900000,
        '30m': 1800000,
        '1H': 3600000,
        '4H': 14400000,
        '1D': 86400000,
        '1W': 604800000
    }
    
    interval_milliseconds = bar_to_milliseconds[bar] * limit
    before_timestamp = start_timestamp
    after_timestamp = before_timestamp + interval_milliseconds
    
    fix = 0
    while after_timestamp <= end_timestamp:
        print("Fetching data between {} and {}".format(pd.Timestamp(after_timestamp, unit='ms'), pd.Timestamp(before_timestamp, unit='ms')))
        result = market.get_history_index_candles(instId=instId, bar=bar, limit=str(limit), before=before_timestamp, after=after_timestamp)
        # print(result)
        if result and 'data' in result:
            all_data.extend(result['data'])
            # print(all_data)
        else:
            print("Failed to fetch data for the current interval.")
        print(result['data'])
        after_timestamp += interval_milliseconds
        before_timestamp += interval_milliseconds

        # If after_timestamp exceeds end_timestamp, adjust it
        if after_timestamp >= end_timestamp:
            after_timestamp = end_timestamp -1
            fix = fix + 1
        
        if fix > 1:
            break
        time.sleep(0.1)  # To avoid hitting API rate limits
    
    return all_data

def convert_to_dataframe(result):
    """Convert the data into a DataFrame."""
    df = pd.DataFrame(result, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close']]
    df['Timestamp'] = df['Timestamp'].astype(int)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    return df

if __name__ == '__main__':
    instId='BTC-USDT'
    start_date='2023-09-01'
    end_date='2023-10-19'
    bar='5m'
    data = fetch_data_between_dates(instId, start_date, end_date, bar)
    df = convert_to_dataframe(data)
    # Sort the data by 'Timestamp'
    df_sorted = df.sort_values(by='Timestamp')
    df_sorted.to_csv(f"{instId}-from-{start_date}-to-{end_date}.csv", index=False)
