import math
import sys
import os
import pytz

from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, GARMIN_HR

functions = {
    "HR2": {
        "name": "get_change_point_hr",
        "description": "Detects change points in heart rate data for a user within a specified time range.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to detect change points in heart rate data for a user within a specified time range.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "int or str", "description": "Start timestamp in seconds or a string in 'YYYY-MM-DD HH:MM:SS' format"},
            "end_time": {"type": "int or str", "description": "End timestamp in seconds or a string in 'YYYY-MM-DD HH:MM:SS' format"}
        },
        "returns": "A list of change points detected in the heart rate data (list of timestamps)",
        "example": " [{'start_time': '2025-02-19 08:23:07', 'end_time': '2025-02-19 08:24:12', 'average_heart_rate': 85.4}, {'start_time': '2025-02-19 08:24:14', 'end_time': '2025-02-19 08:33:54', 'average_heart_rate': 105.9075907590759}, {'start_time': '2025-02-19 08:33:56', 'end_time': '2025-02-19 08:34:57', 'average_heart_rate': 117.39393939393939}]"
    }
}
def get_garmin_hr_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    hr_records = fetch_records_between_timestamps(uid, start_time, end_time, GARMIN_HR)
    return process_records(uid, hr_records)

def process_records(uid, hr_records):
    unique_data = {record['timestamp']: record for record in hr_records}
    hr_records = list(unique_data.values())
    records = []
    
    for r in hr_records:
        d = {}
        start_time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        d['heart_rate'] = r['heart_rate']
        records.append(d)
    
    return records

def mean(values):
    return sum(values) / len(values)

def std_dev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def get_change_point_hr(uid, start_time, end_time, threshold_z=2.0, min_segment_duration=60, min_hr_diff=10):
    hr_records = get_garmin_hr_records(uid, start_time, end_time)
    if not hr_records:
        return []

    timestamps = [datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S') for r in hr_records]
    heart_rates = [r['heart_rate'] for r in hr_records]

    change_points = []
    current_segment = [heart_rates[0]]
    current_start_time = timestamps[0]

    for i in range(1, len(heart_rates)):
        current_segment.append(heart_rates[i])

        # Rolling window of 6 points
        window = heart_rates[max(0, i - 5): i + 1]
        if len(window) < 2:
            continue

        mean_hr = mean(window)
        std_hr = std_dev(window)
        if std_hr == 0:
            continue

        z_score = abs((heart_rates[i] - mean_hr) / std_hr)
        duration = (timestamps[i] - current_start_time).total_seconds()

        if z_score > threshold_z and duration >= min_segment_duration:
            prev_avg = mean(current_segment[:-1])
            curr_hr = heart_rates[i]

            # Check heart rate difference
            if abs(curr_hr - prev_avg) >= min_hr_diff:
                change_points.append({
                    'start_time': current_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': timestamps[i - 1].strftime('%Y-%m-%d %H:%M:%S'),
                    'average_heart_rate': prev_avg
                })
                current_start_time = timestamps[i]
                current_segment = [heart_rates[i]]

    # Add final segment
    if current_segment:
        change_points.append({
            'start_time': current_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': timestamps[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'average_heart_rate': mean(current_segment)
        })

    return change_points


if __name__ == "__main__":
    uid = "pilot2"
    start_time = "2025-02-19 11:46:00"
    end_time = "2025-02-19 21:55:00"

    # Example usage
    hr_records = get_garmin_hr_records(uid, start_time, end_time)
    print("Heart Rate Records:", hr_records)
