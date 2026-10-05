import sys
import os
import pytz

from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

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

if __name__ == "__main__":
    print(get_pixel_skin_temp_records("pilot2", "2025-02-19 11:00:00", "2025-02-19 12:00:00"))
