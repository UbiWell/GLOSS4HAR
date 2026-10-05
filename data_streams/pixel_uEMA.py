import math
import sys
import os
import pytz

from datetime import datetime, timedelta


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

import agents.generic_summarizer
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, UEMA

functions = {
    "UEMA1": {
        "name": "get_pixel_uema_records",
        "description": "Fetches pixel uema records for a user between specified timestamps.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to get pixel uema records for a user between specified timestamps.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A list of user self-reported in-the-moment activity, with timestamp and the self-report",
        "example": "[{'timestamp': '2025-02-19 08:00:00', 'self_report': 'I am working on a project'}, {'timestamp': '2025-02-19 08:00:00', 'self_report': 'I am sitting and relaxing'}]"
    },
    "UEMA2": {
        "name": "get_pixel_uEMA_records",
        "description": "Fetches pixel uema records for a user between specified timestamps.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to get pixel uema records for a user between specified timestamps.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A list of user self-reported in-the-moment activity, with timestamp and the self-report",
        "example": "[{'timestamp': '2025-02-19 08:00:00', 'self_report': 'I am working on a project'}, {'timestamp': '2025-02-19 08:00:00', 'self_report': 'I am sitting and relaxing'}]"
    }
}

def get_pixel_uema_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    uema_records = fetch_records_between_timestamps(uid, start_time, end_time, UEMA)
    return process_uema_records(uema_records)

# Same function under the name listed as UEMA2 in `functions` above
get_pixel_uEMA_records = get_pixel_uema_records

def process_uema_records(uema_records):
    records = []
    for record in uema_records:
        d = {}
        d['timestamp'] = datetime.fromtimestamp(record['timestamp'], tz = pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
        d['self_report'] = record['uEMA']
        records.append(d)

    # remove duplicates
    seen = set()
    records = [x for x in records if not (tuple(x.items()) in seen or seen.add(tuple(x.items())))]
    # sort by timestamp
    records.sort(key=lambda x: x['timestamp'])
    
    return records


if __name__ == "__main__":
    print(get_pixel_uema_records("pilot2", "2025-02-19 11:00:00", "2025-02-19 12:00:00"))
