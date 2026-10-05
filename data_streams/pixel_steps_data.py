import sys
import os
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, PIXEL_STEPS

functions = {
    "STEP2": {
        "name": "detect_step_periods_within_time_range",
        "usecase": ["code_generation"],
        "description": "Detects periods of time where the user was actively taking steps within a specified time range.",
        "function_call_instructions": "only call this function if you need to check for periods of time where the user was actively taking steps",
        "params": {
            "uid": {"type": "string",
                    "description": "The unique identifier for the user whose steps data is to be analyzed."},
            "start_time": {"type": "string",
                           "description": "The start of the time range for which steps data is to be analyzed."},
            "end_time": {"type": "string",
                         "description": "The end of the time range for which steps data is to be analyzed."},
        },
        "returns": "A list of periods where the user was actively taking steps, including start time, end time, average steps per minute, and total steps.",
        "example": "[{'start_time': '2024-07-20 10:00:00', 'end_time': '2024-07-20 10:30:00', 'average_steps_per_minute': 120}, {'start_time': '2024-07-20 15:00:00', 'end_time': '2024-07-20 15:45:00', 'average_steps_per_minute': 100}]"
    }
}

def get_steps_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    steps_records = fetch_records_between_timestamps(uid, start_time, end_time, PIXEL_STEPS)
    return process_records(uid, steps_records)

def process_records(uid, step_records):
    unique_data = {record['timestamp']: record for record in step_records}
    step_records = list(unique_data.values())
    records = []
    for r in step_records:
        d = {}
        start_time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        d['steps'] = r['steps']
        records.append(d)
    return records

def detect_step_periods_within_time_range(uid, start_time, end_time):
    steps_records =  get_steps_records(uid, start_time, end_time)
    if len(steps_records) == 0:
        return "No steps recorded during this period"
    
    # sort by timestamp
    steps_records.sort(key=lambda x: x['timestamp'])
    # we want to find periods of time where steps are greater than 0
    periods = []
    current_period = []
    for record in steps_records:
        if record['steps'] > 0:
            current_period.append(record)
        else:
            if current_period:
                periods.append(current_period)
                current_period = []
    if current_period:
        periods.append(current_period)
    if not periods:
        return []

    # return period as {start_time, end_time, average_steps_per_minute, total_steps}
    result = []
    for period in periods:
        start_time = period[0]['timestamp']
        end_time = period[-1]['timestamp']
        duration_minutes = (datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')).total_seconds() / 60
        total_steps = sum(record['steps'] for record in period)
        average_steps_per_minute = total_steps / duration_minutes if duration_minutes > 0 else total_steps
        result.append({
            'start_time': start_time,
            'end_time': end_time,
            'average_steps_per_minute': round(average_steps_per_minute, 2),
        })
    return result

if __name__ == "__main__":
    uid = "pilot7"
    start_time = "2025-02-27 11:30:00"
    end_time = "2025-02-27 12:00:00"
    
    # Example usage
    records = detect_step_periods_within_time_range(uid, start_time, end_time)
    for record in records:
        print(f"Start Time: {record['start_time']}, End Time: {record['end_time']}, Average Steps per Minute: {record['average_steps_per_minute']}")