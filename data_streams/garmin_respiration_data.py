import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import GARMIN_IBI

def get_garmin_ibi(uid, start_time, end_time):
    ibi_records = fetch_documents_between_timestamps(uid, start_time, end_time, GARMIN_IBI)
    return ibi_records

def count_ibi_values(ibi_records):
    ibi_values = [entry['bbi'] for entry in ibi_records]
    ibi_counts = dict(Counter(ibi_values))
    return ibi_counts

def plot_ibi(ibi_records):
    timestamps = []
    ibi_values = []
    for entry in ibi_records:
        if entry['bbi'] != -99:
            timestamps.append(datetime.fromtimestamp(entry['timestamp']))
            ibi_values.append(entry['bbi'])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, ibi_values)
    plt.xlabel('Timestamp')
    plt.ylabel('Interbeat Interval (ms)')
    plt.title('Interbeat Interval vs Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    start_datetime = datetime(2024, 6, 25, 0, 0, 0)
    end_datetime = datetime(2024, 6, 25, 12, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    ibi_records = get_garmin_ibi('test004', start_timestamp, end_timestamp)
    print(count_ibi_values(ibi_records))
    plot_ibi(ibi_records)
