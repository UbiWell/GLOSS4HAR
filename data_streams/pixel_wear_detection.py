import sys
import os
import pytz

from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, PIXEL_WEAR_DETECTION

functions = {
    "WEAR_DETECTION2": {
        "name": "detect_non_wear_period_within_time_range",
        "description": "Detects periods of non-wear within a specified time range for a user.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to detect periods of non-wear within a specified time range for a user.",
        "params": {
            "uid": {"type": "string", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A list of non-wear periods with start and end timestamps, or a string indicating continuous wear.",
        "example": "[{'start_time': '2024-07-20 10:00:00', 'end_time': '2024-07-20 10:30:00'}, {'start_time': '2024-07-20 15:00:00', 'end_time': '2024-07-20 15:45:00'}]",
        "example_continuous_wear": "User has worn the device continuously in the given time range."
    }
}

def get_pixel_wear_detection_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    wear_detection_records = fetch_records_between_timestamps(uid, start_time, end_time, PIXEL_WEAR_DETECTION)
    return process_records(uid, wear_detection_records)

def process_records(uid, wear_detection_records):
    unique_data = {record['timestamp']: record for record in wear_detection_records}
    wear_detection_records = list(unique_data.values())
    records = []
    
    for r in wear_detection_records:
        d = {}
        start_time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        # if wear_detection is True -> Worn, else Not Worn
        d['wear_status'] = 'Worn' if r['wear_detection'] else 'Not Worn'
        # Add the processed record to the list
        records.append(d)
    
    return records

def detect_non_wear_period_within_time_range(uid, start_time, end_time):
    wear_detection_records = get_pixel_wear_detection_records(uid, start_time, end_time)
    if len(wear_detection_records) == 0:
        return []
    non_wear_periods = []
    current_period = None
    #sort by timestamp
    wear_detection_records.sort(key=lambda x: x['timestamp'])
    # check if there are consecutive records with wear_status 'Not Worn'
    for record in wear_detection_records:
        if record['wear_status'] == 'Not Worn':
            if current_period is None:
                current_period = {
                    'start_time': record['timestamp'],
                    'end_time': record['timestamp']
                }
            else:
                current_period['end_time'] = record['timestamp']
        else:
            if current_period is not None:
                non_wear_periods.append(current_period)
                current_period = None
    if current_period is not None:
        non_wear_periods.append(current_period)
    if len(non_wear_periods) == 0:
        print("User has worn the device continuously in the given time range.")
        return {}
    return non_wear_periods