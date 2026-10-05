import sys
import os
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime, timedelta
from data_processing.data_processing_utils import fetch_records_between_timestamps
from data_streams.constants import ANDROID_LOCATION, time_zone_dict
from geopy import distance

functions = {
    "LOC1": {
        "name": "get_android_location_records",
        "description": "Retrieves Android location records for a user between specified timestamps.",
        "usecase": ["code_generation"],
        "params": {
            "uid": {"type": "str", "description": "The unique identifier for the user."},
            "start_time": {"type": "str", "description": "The start time of the period."},
            "end_time": {"type": "str", "description": "The end time of the period."}
        },
        "returns": {
            "location_records": {"type": "list", "description": "A list of location records with latitude, longitude, and timestamp."}
        },
        "example": "[{'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': '2024-07-20 00:01:00'}, {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': '2024-07-20 00:02:00'}]"
    },
        "LOC3": {
        "name": "get_location_paths",
        "description": "Extracts and organizes distinct paths taken by a user based on GPS location data within a specified time range.",
        "usecase": ["code_generation", "function_calling"],
        "params": {
            "uid": {"type": "str",
                    "description": "The unique identifier for the user whose location paths are to be retrieved."},
            "start_time": {"type": "str",
                           "description": "The start of the time range for which location data is to be analyzed."},
            "end_time": {"type": "str",
                         "description": "The end of the time range for which location data is to be analyzed."}
        },
        "returns":  "GPS location of starting and end point of paths taken",
        "example": "[{'starting_point': {'timestamp': 1720541549, 'latitude': 40.712776, 'longitude': -74.005974, 'altitude': 15.27495}, 'end_point': {'timestamp': 1720541549, 'latitude': 40.712776, 'longitude': -74.005974, 'altitude': 15.27495}}, {'starting_point': {'timestamp': 1720565225, 'latitude': 40.714268, 'longitude': -74.003291, 'altitude': 13.192509}, 'end_point': {'timestamp': 1720566122, 'latitude': 40.722391, 'longitude': -74.001542, 'altitude': 6.118234}}]"
    },
}

def get_android_location_records(uid, start_time, end_time, select_one_from_minute=False):
    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    location_log = fetch_records_between_timestamps(uid, start_time, end_time, ANDROID_LOCATION)
    if not location_log:
        return []
    return process_records(uid, location_log)

def process_records(uid, location_records):
    records = []
    for r in location_records:
        d = {}
        time = datetime.fromtimestamp(r['timestamp'], tz = pytz.timezone('US/Eastern'))

        # Format the timestamp and add it to the record
        d['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        d['latitude'] = r['latitude']
        d['longitude'] = r['longitude']
        records.append(d)
    return records

def get_location_paths(uid, start_time, end_time):
    coords = get_android_location_records(uid, start_time, end_time, True)
    paths = []
    current_path = []
    last_moving_coord = coords[0]
    places_visited = []
    places_tracks = []

    # Convert the timestamp to Unix time
    for coord in coords:
        coord['timestamp'] = int(datetime.strptime(coord['timestamp'], '%Y-%m-%d %H:%M:%S').timestamp())

    for i in range(len(coords)):
        if i == 0 or get_distance([coords[i]['latitude'], coords[i]['longitude']],
                                  [coords[i - 1]['latitude'], coords[i - 1]['longitude']]) > 100:

            # Check if there is a gap of 10 minutes or more between current and last moving coordinates
            if (coords[i]['timestamp'] - last_moving_coord['timestamp'] >= 10 * 60):
                paths.append(current_path)
                places_visited.append(last_moving_coord)
                current_path = []

            current_path += [coords[i]]
            last_moving_coord = coords[i]

    # Add the last path if any
    if current_path:
        paths.append(current_path)

    user_timezone = time_zone_dict.get(uid, "UTC")

    for path in paths:
        if(len(path) > 1):
            duration = path[-1]['timestamp'] - path[0]['timestamp']
            path[0]['timestamp'] = datetime.fromtimestamp(path[0]['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            path[-1]['timestamp'] = datetime.fromtimestamp(path[-1]['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

            places_tracks.append({'starting_point': path[0], 'end_point': path[-1], 'duration': duration})

    return places_tracks

def get_distance(loc0, loc1):
    return distance.distance(loc0, loc1).m

if __name__ == "__main__":
    print(get_android_location_records("pilot2", "2025-02-19 10:58:00", "2025-02-19 11:24:00"))
