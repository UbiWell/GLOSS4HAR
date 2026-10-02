import sys
import os

from data_processing.plotting_utils import plot_blocks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import IOS_BLUETOOTH


def get_bluetooth_metrics(uid, start_time, end_time):
    bluetooth_records = fetch_documents_between_timestamps(uid, start_time, end_time, IOS_BLUETOOTH)
    return bluetooth_records

def get_bluetooth_blocks(bluetooth_records):
    if not bluetooth_records:
        return []

    start_device = bluetooth_records[0]['bt_name']
    start_time = bluetooth_records[0]['timestamp']
    bluetooth_blocks = []
    for i in range(1, len(bluetooth_records)):
        if bluetooth_records[i]['bt_name'] != bluetooth_records[i - 1]['bt_name']:
            bluetooth_blocks.append([start_device, start_time, bluetooth_records[i]['timestamp']])
            start_device = bluetooth_records[i]['bt_name']
            start_time = bluetooth_records[i]['timestamp']
    # Ensure the last block is added
    bluetooth_blocks.append([start_device, start_time, bluetooth_records[-1]['timestamp']])
    return bluetooth_blocks

def generate_bluetooth_summary(activity_blocks):
    total_time = {}
    for entry in activity_blocks:
        bt = entry[0]
        start_time = entry[1]
        end_time = entry[2]

        duration = end_time - start_time

        if bt in total_time:
            total_time[bt] += duration / (60)
        else:
            total_time[bt] = duration / (60)
    return total_time



if __name__ == "__main__":
    start_datetime = datetime(2024, 7, 9, 0, 0, 0)
    end_datetime = datetime(2024, 7, 11, 23, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    bluetooth_records = get_bluetooth_metrics('test004', start_timestamp, end_timestamp)
    bluetooth_activity_blocks = get_bluetooth_blocks(bluetooth_records)
    print(bluetooth_activity_blocks)
    print(generate_bluetooth_summary(bluetooth_activity_blocks))