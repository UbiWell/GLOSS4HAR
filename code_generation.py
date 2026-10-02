# filename: code_generation.py
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from geopy.distance import great_circle
from scipy import spatial
from geopy import distance
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import *
import folium
from geopy.geocoders import Nominatim, GoogleV3
from agents.coding_agent import run_coding_agent

# Import data stream functions
from data_streams.android_phone_usage import get_phone_usage_period
from data_streams.pixel_steps_data import detect_step_periods_within_time_range
from data_streams.pixel_hr import get_change_point_hr
from data_streams.pixel_wear_detection import detect_non_wear_period_within_time_range
from data_streams.android_location import get_android_location_records, get_location_paths

# Define user and time range
uid = "pilot2"
start_time = "2025-02-19 22:43:00"
end_time   = "2025-02-19 23:23:00"

def main():
    print("Fetching phone usage data...")
    try:
        phone_usage = get_phone_usage_period(uid, start_time, end_time)
        print("Phone usage periods:")
        print(phone_usage)
    except Exception as e:
        print(f"Error fetching phone usage data: {e}")

    print("\nFetching step count data...")
    try:
        steps = detect_step_periods_within_time_range(uid, start_time, end_time)
        print("Step periods:")
        print(steps)
    except Exception as e:
        print(f"Error fetching step count data: {e}")

    print("\nFetching heart rate data...")
    try:
        heart_rate_changes = get_change_point_hr(uid, start_time, end_time)
        print("Heart rate change points:")
        print(heart_rate_changes)
    except Exception as e:
        print(f"Error fetching heart rate data: {e}")

    print("\nFetching watch wear status data...")
    try:
        non_wear_periods = detect_non_wear_period_within_time_range(uid, start_time, end_time)
        print("Non‑wear periods:")
        print(non_wear_periods)
    except Exception as e:
        print(f"Error fetching watch wear data: {e}")

    print("\nFetching location data (records)...")
    try:
        location_records = get_android_location_records(uid, start_time, end_time)
        print("Location records:")
        print(location_records)
    except Exception as e:
        print(f"Error fetching location data: {e}")

    print("\nFetching location path summaries...")
    try:
        location_paths = get_location_paths(uid, start_time, end_time)
        print("Location paths:")
        print(location_paths)
    except Exception as e:
        print(f"Error fetching location path data: {e}")

    print("\nAll data retrieval completed.")

if __name__ == "__main__":
    main()
    print("\nTERMINATE")
