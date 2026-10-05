import os
from functools import lru_cache

import pandas as pd

# Folder holding one CSV per data stream (e.g. android_location.csv, uEMA.csv).
# Override with the GLOSS4HAR_DATA_DIR environment variable.
DATA_DIR = os.getenv(
    "GLOSS4HAR_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
)


@lru_cache(maxsize=None)
def _load_stream(stream_name):
    path_to_csv_file = os.path.join(DATA_DIR, f"{stream_name}.csv")
    if not os.path.exists(path_to_csv_file):
        raise FileNotFoundError(
            f"No data file for stream '{stream_name}': expected {path_to_csv_file}. "
            f"Set GLOSS4HAR_DATA_DIR to the folder containing the stream CSVs."
        )
    return pd.read_csv(path_to_csv_file)


def fetch_records_between_timestamps(uid, start_timestamp, end_timestamp, stream_name):
    """
    Fetch records of one data stream for a user between two timestamps.

    Parameters:
    - uid (str): The subject id (matches the `subject_id` column).
    - start_timestamp (float): Start time, epoch seconds.
    - end_timestamp (float): End time, epoch seconds.
    - stream_name (str): Name of the data stream; read from <DATA_DIR>/<stream_name>.csv.

    Returns:
    - list: Records (dicts) sorted by timestamp, with `timestamp` in epoch seconds.
    """
    df = _load_stream(stream_name)
    # the csv stores timestamps in epoch milliseconds
    start_timestamp = int(start_timestamp * 1000)
    end_timestamp = int(end_timestamp * 1000)
    filtered_df = df[(df['subject_id'] == uid) & (df['timestamp'] >= start_timestamp) & (df['timestamp'] < end_timestamp)]
    filtered_df = filtered_df.sort_values(by='timestamp')
    filtered_df['timestamp'] = filtered_df['timestamp'] / 1000
    return filtered_df.to_dict('records')


if __name__ == "__main__":
    from datetime import datetime

    start_timestamp = datetime(2025, 2, 18, 0, 0, 0).timestamp()
    end_timestamp = datetime(2025, 2, 20, 23, 59, 59).timestamp()

    for doc in fetch_records_between_timestamps("pilot2", start_timestamp, end_timestamp, 'android_location'):
        print(doc)
