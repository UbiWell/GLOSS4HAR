import math
import sys
import os
import pytz

from datetime import datetime, timedelta
from agents.coding_agent import run_coding_agent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

import agents.generic_summarizer
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import time_zone_dict, PIXEL_WRIST_AUC

functions = {
    "WRIST_AUC1": {
        "name": "get_pixel_wrist_auc_records",
        "description": "Fetches wrist AUC records for a user between specified timestamps.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to get wrist AUC records for a user between specified timestamps.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A list of wrist AUC records with timestamp and the AUC value",
        "example": "[{'timestamp': '2024-07-20 00:15:08', 'wrist_auc': 12.3}, {'timestamp': '2024-07-20 00:16:15', 'wrist_auc': 120.34}, {'timestamp': '2024-07-20 02:55:33', 'wrist_auc': 250.3}]"
    },
    "WRIST_AUC2": {
        "name": "detect_change_point_auc",
        "description": "Detects change points in wrist AUC data for a user within a specified time range.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to detect change points in wrist AUC data for a user within a specified time range.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "str", "description": "Start timestamp in seconds or a string in 'YYYY-MM-DD HH:MM:SS' format"},
            "end_time": {"type": "str", "description": "End timestamp in seconds or a string in 'YYYY-MM-DD HH:MM:SS' format"}
        },
        "returns": "A list of change points detected in the wrist AUC data (list of timestamps)",
        "example": "['2024-07-20 00:15:08', '2024-07-20 00:16:15']"
    }
}

def get_pixel_wrist_auc_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    wrist_auc_records = fetch_documents_between_timestamps(uid, start_time, end_time, PIXEL_WRIST_AUC)
    return process_records(uid, wrist_auc_records)

def process_records(uid, wrist_auc_records):
    unique_data = {record['timestamp']: record for record in wrist_auc_records}
    wrist_auc_records = list(unique_data.values())
    records = []
    uid_timezone = time_zone_dict.get(uid, 'utc')  # Get UID-specific timezone or default to UTC
    timezone = pytz.timezone("America/New_York") if uid_timezone == "est" else pytz.utc
    
    for r in wrist_auc_records:
        d = {}
        start_time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        d['wrist_auc'] = r['total_auc']
        records.append(d)
    
    return records

def detect_change_point_auc(uid, start_time, end_time):
    '''
    Detects change points in wrist AUC data for a user within a specified time range.
    Args:
        uid (str): User ID.
        start_time (int or str): Start timestamp in seconds or a string in "YYYY-MM-DD HH:MM:SS" format.
        end_time (int or str): End timestamp in seconds or a string in "YYYY-MM-DD HH:MM:SS"
    Returns:
        list: A list of change points detected in the wrist AUC data (list of timestamp).
    '''
    records = get_pixel_wrist_auc_records(uid, start_time, end_time)
    
    if not records:
        return []
    
    timestamps = [record['timestamp'] for record in records]
    auc_values = [record['wrist_auc'] for record in records]
    change_points = []

    # Calculate the first derivative of the AUC values
    first_derivative = [auc_values[i] - auc_values[i - 1] for i in range(1, len(auc_values))]
    # Calculate the second derivative of the AUC values
    second_derivative = [first_derivative[i] - first_derivative[i - 1] for i in range(1, len(first_derivative))]
    # Detect change points where the second derivative is significantly different from zero
    threshold = 0.05 * (1000 - 30)  # Define a threshold for detecting significant changes
    for i in range(1, len(second_derivative)):
        if abs(second_derivative[i]) > threshold:
            change_points.append(timestamps[i + 1])  # +1 because second_derivative is shorter by 2
    return change_points