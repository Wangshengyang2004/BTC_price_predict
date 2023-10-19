from pymongo import MongoClient
from okx_api import Market
import pandas as pd
import time



def date_to_timestamp(date_str):
    """Convert a date string to a timestamp."""
    return int(pd.Timestamp(date_str).timestamp() * 1000)

def get_latest_timestamp(collection_name):
    """Get the latest timestamp from a MongoDB collection."""
    client = MongoClient('192.168.31.247', 27017)  # Connect to MongoDB running on localhost
    db = client[instId]  # Database named 'BTC-USDT'
    collection = db[collection_name]
    latest_entry = collection.find_one(sort=[("Timestamp", -1)])
    if latest_entry:
        return latest_entry['Timestamp']
    return None

def fetch_data_between_dates(instId, start_date, end_date, bar, limit=100):
    """Fetch historical data between start_date and end_date."""
    
    market = Market()
    all_data = []
    
    # Convert start_date and end_date to timestamps
    start_timestamp = date_to_timestamp(start_date)
    end_timestamp = date_to_timestamp(end_date)
    
    collection_name = bar
    latest_timestamp_in_db = get_latest_timestamp(collection_name)
    if latest_timestamp_in_db:
        start_timestamp = latest_timestamp_in_db + 1  # Update the start_timestamp to fetch only missing data
        
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
    df['Timestamp'] = df['Timestamp'].astype(int)
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    return df

from pymongo import MongoClient

def update_mongodb(df, bar):
    # Connect to MongoDB
    # MongoDB setup
    client = MongoClient('192.168.31.247', 27017)  # Connect to MongoDB running on localhost
    db = client[instId]  # Database named 'BTC-USDT'
    collection = db[bar]

    # Convert the DataFrame to a list of dictionaries
    records = df.to_dict(orient='records')

    # Insert the data into MongoDB
    collection.insert_many(records)


if __name__ == '__main__':
    instId='TRB-USDT'
    start_date='2023-01-01'
    end_date='2023-10-19'
    bar='15m'
    data = fetch_data_between_dates(instId, start_date, end_date, bar)
    df = convert_to_dataframe(data)
    # Sort the data by 'Timestamp'
    df_sorted = df.sort_values(by='Timestamp')
    df_sorted.to_csv(f"{instId}-from-{start_date}-to-{end_date}.csv", index=False)
    
    # Update the MongoDB database
    update_mongodb(df_sorted, bar)
