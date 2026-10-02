import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import GARMIN_ENERGY
import matplotlib.pyplot as plt


def get_garmin_energy_records(uid, start_time, end_time):
    energy_records = fetch_documents_between_timestamps(uid, start_time, end_time, GARMIN_ENERGY)
    return energy_records



if __name__ == "__main__":
    start_datetime = datetime(2024, 6, 25, 12, 0, 0)
    end_datetime = datetime(2024, 6, 26, 18, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    energy_records = get_garmin_energy_records('test004', start_timestamp, end_timestamp)
    print(energy_records)