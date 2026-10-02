import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import EMPATICA_EDA

def get_garmin_eda(uid, start_time, end_time):
    eda_records = fetch_documents_between_timestamps(uid, start_time, end_time, EMPATICA_EDA)
    return eda_records

def count_eda_values(eda_records):
    eda_values = [entry['eda'] for entry in eda_records]
    eda_counts = dict(Counter(eda_values))
    return eda_counts

def plot_eda(eda_records):
    timestamps = []
    eda_values = []
    for entry in eda_records:
        if entry['eda'] != -99:
            timestamps.append(datetime.fromtimestamp(entry['timestamp']))
            eda_values.append(entry['eda'])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, eda_values)
    plt.xlabel('Timestamp')
    plt.ylabel('Electrodermal Activity (EDA)')
    plt.title('Electrodermal Activity vs Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_datetime = datetime(2024, 6, 25, 0, 0, 0)
    end_datetime = datetime(2024, 6, 25, 12, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    eda_records = get_garmin_eda('test004', start_timestamp, end_timestamp)
    print(count_eda_values(eda_records))
    # plot_eda(eda_records)
