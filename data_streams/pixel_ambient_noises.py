import math
import sys
import os
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
from datetime import datetime, timedelta
from agents.coding_agent import run_coding_agent
import agents.generic_summarizer
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import time_zone_dict, PIXEL_AMBIENT_NOISE

functions = {
    "NOISE1": {
        "name": "get_pixel_ambient_noise_records",
        "description": "Fetches ambient noise records for a user between specified timestamps.",
        "usecase": ["code_generation"],
        "function_call_instructions": "Call this function to get ambient noise records for a user between specified timestamps.",
        "params": {
            "uid": {"type": "str", "description": "User ID"},
            "start_time": {"type": "string", "description": "Start timestamp in seconds"},
            "end_time": {"type": "string", "description": "End timestamp in seconds"}
        },
        "returns": "A list of ambient noise records with timestamp and the noise level",
        "example": "[{'timestamp': '2024-07-20 00:15:08', 'ambient_noise': 'Ambience Noise:[Silence]'}, {'timestamp': '2024-07-20 00:16:15', 'ambient_noise': 'Ambience Noise:[Speech]'}, {'timestamp': '2024-07-20 02:55:33', 'ambient_noise': 'Ambience Noise:[Speech, Inside, small room]'}]"
    }
}

def get_pixel_ambient_noise_records(uid, start_time, end_time):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    ambient_noise_records = fetch_records_between_timestamps(uid, start_time, end_time, PIXEL_AMBIENT_NOISE)
    return process_records(uid, ambient_noise_records)

def process_records(uid, ambient_noise_records):
    unique_data = {record['timestamp']: record for record in ambient_noise_records}
    ambient_noise_records = list(unique_data.values())
    records = []
    
    for r in ambient_noise_records:
        d = {}
        start_time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))
        d['timestamp'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        d['ambient_noise'] = r['ambient_noise']
        records.append(d)
    
    return records

if __name__ == "__main__":
    uid = "pilot2"
    start_time = "2025-02-19 8:05:00"
    end_time = "2025-02-19 8:25:00"

    # Example usage
    records = get_pixel_ambient_noise_records(uid, start_time, end_time)
    print(records)