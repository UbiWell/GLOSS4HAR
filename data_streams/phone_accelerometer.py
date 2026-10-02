import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import IOS_ACCELEROMETER
import matplotlib.pyplot as plt


def get_accelerometer_records(uid, start_time, end_time):
    acc_records = fetch_documents_between_timestamps(uid, start_time, end_time, IOS_ACCELEROMETER)
    return acc_records



if __name__ == "__main__":
    start_datetime = datetime(2024, 7, 9, 12, 0, 0)
    end_datetime = datetime(2024, 7, 9, 18, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    acc_records = get_accelerometer_records('test004', start_timestamp, end_timestamp)
    print(acc_records)
