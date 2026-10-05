import math
import sys
import os
import pytz

from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
from agents.coding_agent import run_coding_agent

import agents.generic_summarizer
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, ANDROID_PHONE_USAGE

functions = {
    # "USAGE1": {
    #     "name": "get_phone_usage_records",
    #     "description": "Fetches phone usage records for a user between specified timestamps.",
    #     "usecase": ["code_generation"],
    #     "function_call_instructions": "Call this function to get phone usage records for a user between specified timestamps.",
    #     "params": {
    #         "uid": {"type": "str", "description": "User ID"},
    #         "start_time": {"type": "string", "description": "Start timestamp in seconds"},
    #         "end_time": {"type": "string", "description": "End timestamp in seconds"}
    #     },
    #     "returns": "A list of phone usage records, with True/False values indicating whether the phone was in use or not, along with timestamps.",
    #     "example": "[{'timestamp': 1720541549, 'in_use': 'True'}, {'timestamp': 1720565225, 'in_use': 'False'}]"
    # }
    # ,
    "USAGE2": {
        "name": "get_phone_usage_period",
        "description": "Summarizes phone usage periods for a user between specified timestamps.",
        "usecase": ["code_generation", "summarization"],
        "function_call_instructions": "Call this function to get a summary of phone usage periods for a user between specified timestamps.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A summary of phone usage periods, indicating when the phone was in use and when it was not.",
        "example": "\"Phone usage periods:\\nFrom 2025-03-10 08:00:00 to 2025-03-10 08:30:00\\nFrom 2025-03-10 09:00:00 to 2025-03-10 09:15:00\""
    }
}

def get_phone_usage_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    usage_records = fetch_records_between_timestamps(uid, start_time, end_time, ANDROID_PHONE_USAGE)
    return process_usage_records(usage_records)

def process_usage_records(usage_records):
    records = []
    for record in usage_records:
        d = {}
        d['timestamp'] = datetime.fromtimestamp(record['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = d['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        d['in_use'] = str(record['in_use'])
        records.append(d)
    return records

def get_phone_usage_period(uid, start_time, end_time):
    usage_records = get_phone_usage_records(uid, start_time, end_time)
    if not usage_records:
        return "No phone usage records found in the specified time range."

    usage_periods = []
    current_period = None

    # remove the seconds from the timestamp
    for record in usage_records:
        # timestamp in format HH:MM:SS
        record['timestamp'] = record['timestamp'].split(' ')[0] + ' ' + record['timestamp'].split(' ')[1].split(':')[0] + ':' + record['timestamp'].split(' ')[1].split(':')[1]

    for record in usage_records:
        if record['in_use'] == 'True':
            if current_period is None:
                current_period = {'start': record['timestamp'], 'end': record['timestamp']}
            else:
                current_period['end'] = record['timestamp']
        else:
            if current_period is not None:
                usage_periods.append(current_period)
                current_period = None

    if current_period is not None:
        usage_periods.append(current_period)

    if not usage_periods:
        return "No phone usage periods found in the specified time range."

    summary = "Phone usage periods:\n"
    for period in usage_periods:
        if period['start'] == period['end']:
            summary += f"Less than a minute at {period['start'].split(' ')[1]}\n"
        else:
            summary += f"From {period['start'].split(' ')[1]} to {period['end'].split(' ')[1]}\n"

    return summary

if __name__ == "__main__":
    uid = "pilot2"
    start_time = "2025-02-19 11:46:00"
    end_time = "2025-02-19 21:55:00"

    # Example usage
    records = get_phone_usage_records(uid, start_time, end_time)
    print(records)
    # for record in records:
    #     print(f"Timestamp: {record['timestamp']}, In Use: {record['in_use']}")
    #     # print the type of in_use
    #     print(f"Type of In Use: {type(record['in_use'])}")

    # print(get_phone_usage_period(uid, start_time, end_time))