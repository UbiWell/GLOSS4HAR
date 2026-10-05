import math
import sys
import os
import pytz
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
from agents.coding_agent import run_coding_agent

import agents.generic_summarizer
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, PIXEL_SKIN_TEMP

functions = {
    "SKIN_TEMP1": {
        "name": "get_pixel_skin_temp_records",
        "description": "Fetches skin temperature records for a user between specified timestamps.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to get skin temperature records for a user between specified timestamps.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A list of skin temperature records with timestamp and the temperature value",
        "example": "[{'timestamp': '2024-07-20 00:15:08', 'skin_temp': 36.5}, {'timestamp': '2024-07-20 00:16:15', 'skin_temp': 36.7}, {'timestamp': '2024-07-20 02:55:33', 'skin_temp': 36.6}]"
    }
}
def get_pixel_skin_temp_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    skin_temp_records = fetch_records_between_timestamps(uid, start_time, end_time, PIXEL_SKIN_TEMP)
    return process_records(uid, skin_temp_records)

def process_records(uid, skin_temp_records):
    unique_data = {record['timestamp']: record for record in skin_temp_records}
    skin_temp_records = list(unique_data.values())
    records = []
    
    for r in skin_temp_records:
        d = {}
        start_time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        d['skin_temp'] = r['skin_temp']
        records.append(d)
    
    return records

def get_change_point_skin_temp(uid, start_time, end_time):
    '''
    Detecting significant changes in skin temperature over a period of time.
    '''
    skin_temp_records = get_pixel_skin_temp_records(uid, start_time, end_time)

    if not skin_temp_records:
        return []
    skin_temp_values = [record['skin_temp'] for record in skin_temp_records]
    timestamps = [record['timestamp'] for record in skin_temp_records]
    if len(skin_temp_values) < 2:
        return []
    
    smoothing_window = 5  # Number of points to average for smoothing
    z_threshold = 2.0  # Z-score threshold for detecting significant changes
    min_gap_seconds = 60 * 5  # Minimum gap between change points in seconds

    temps = [record['skin_temp'] for record in skin_temp_records]
    # Step 2: Smooth if needed
    if smoothing_window > 1:
        temps = pd.Series(temps).rolling(window=smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').tolist()

    # Step 3: Compute second derivative
    first_derivative = np.diff(temps)
    second_derivative = np.diff(first_derivative)

    # Step 4: Z-score normalization
    if np.std(second_derivative) == 0:
        return []

    z_scores = (second_derivative - np.mean(second_derivative)) / np.std(second_derivative)

    # Step 5: Detect significant changes
    raw_change_points = [timestamps[i + 2] for i, z in enumerate(z_scores) if abs(z) > z_threshold]

    # Step 6: Apply minimum gap filter
    filtered_change_points = []
    for t in raw_change_points:
        if not filtered_change_points or (t - filtered_change_points[-1]).total_seconds() >= min_gap_seconds:
            filtered_change_points.append(t)

    return filtered_change_points
    
    
if __name__ == "__main__":
    uid = "pilot2"
    start_time = "2025-02-19 08:00:00"
    end_time = "2025-02-19 09:00:00"

    skin_temp_records = get_pixel_skin_temp_records(uid, start_time, end_time)
    print(skin_temp_records)
    change_points = get_change_point_skin_temp(uid, start_time, end_time)
    print(change_points)