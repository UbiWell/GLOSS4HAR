import sys
import os
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
from geopy.geocoders import GoogleV3

from datetime import datetime, timedelta
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import ANDROID_LOCATION, time_zone_dict
from agents.coding_agent import run_coding_agent
from data_streams.constants import GOOGLE_API_KEY
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
    #     "LOC2": {
    #     "name": "get_address_from_coordinates",
    #     "usecase": ["code_generation", "function_calling"],
    #     "description": "Retrieves the address of a location given its latitude and longitude.",
    #     "code_generation_instructions": "Do all the processing in latitude and longitude values and only call this function when you need to fetch the address at the last. This is an expensive function to call. Call it at end for few latitude longitude pairs",
    #     "params": {
    #         "latitude": {"type": "float", "description": "The latitude of the location."},
    #         "longitude": {"type": "float", "description": "The longitude of the location."}
    #     },
    #     "returns": "The address of the location if found, otherwise an empty string."
    # },
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
        # "LOC4": {
        # "name": "get_address_at_timestamp",
        # "description": "Fetches the most frequent address at a given time range for a user based on their Android location records.",
        # "usecase": ["code_generation"],
        # "params": {
        #     "uid": {"type": "str", "description": "The unique identifier for the user."},
        #     "start_time": {"type": "str", "description": "The start time of the period in 'YYYY-MM-DD HH:MM:SS' format."},
        #     "end_time": {"type": "str", "description": "The end time of the period in 'YYYY-MM-DD HH:MM:SS' format."}
        # },
        # "returns": {
        #     "most_frequent_address": {"type": "str", "description": "The most frequently visited address within the specified time range."}
        # },
        # "example": "123 Main St, Springfield, USA"
        # }
        # ,
    # "LOC1": {
    #     "name": "get_location_records",
    #     "description": "Retrieves periods of time when users spent at a specific location based on their Android location records.",
    #     "usecase": ["code_generation"],
    #     "params": {
    #         "uid": {"type": "str", "description": "The unique identifier for the user."},
    #         "start_time": {"type": "str", "description": "The start time of the period in 'YYYY-MM-DD HH:MM:SS' format."},
    #         "end_time": {"type": "str", "description": "The end time of the period in 'YYYY-MM-DD HH:MM:SS' format."}
    #     },
    #     "returns": {
    #         "location_change_periods": {"type": "list", "description": "A list of periods when the user spent time at a specific location."}
    #     },
    #     "example": "[{'start_time': '2025-02-19 08:00:03', 'end_time': '2025-02-19 08:30:41', 'address': '123 Main St, Springfield, USA'}"
    # }
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

    location_log = fetch_documents_between_timestamps(uid, start_time, end_time, ANDROID_LOCATION)
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

def get_address_from_coordinates(latitude, longitude):
    def get_place_name(lat, lng, api_key):
        geolocator = GoogleV3(api_key=api_key)
        location = geolocator.reverse((lat, lng), exactly_one=True)
        return location.address if location else ""

    # Convert latitude and longitude to float if they are strings
    if isinstance(latitude, str):
        latitude = float(latitude)
    if isinstance(longitude, str):
        longitude = float(longitude)

    api_key = GOOGLE_API_KEY
    place_name = get_place_name(latitude, longitude, api_key)
    return place_name

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

def get_address_at_timestamp(uid, start_time, end_time):
    '''
    Fetch most frequent address at a given time range for a user.
    This function retrieves the Android location records for a user within a specified time range,
    '''

    location_records = get_android_location_records(uid, start_time, end_time)
    if not location_records:
        return "No location records found for the specified time range."
    # Convert the timestamps to datetime objects
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    # Filter the records to only include those within the specified time range
    filtered_records = [record for record in location_records if start_time <= datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S") <= end_time]
    if not filtered_records:
        return "No location records found for the specified time range."
    # cluster the records by address
    address_count = {}
    for record in filtered_records:
        address = get_address_from_coordinates(record['latitude'], record['longitude'])
        if address:
            if address not in address_count:
                address_count[address] = 0
            address_count[address] += 1
    # Find the address with the highest count
    if not address_count:
        return "No addresses found for the specified time range."
    most_frequent_address = max(address_count, key=address_count.get)
    return most_frequent_address

def get_location_records(uid, start_time, end_time):
    """
    Get the periods of time when users stay at the same location.
    """
    location_records = get_android_location_records(uid, start_time, end_time)
    if not location_records:
        return "No location records found for the specified time range."
    # Convert the timestamps to datetime objects
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    # Filter the records to only include those within the specified time range
    filtered_records = [record for record in location_records if start_time <= datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S") <= end_time]
    if not filtered_records:
        return "No location records found for the specified time range."
    # Sort the records by timestamp
    filtered_records.sort(key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"))
    # Find periods of time when users stay at the same location
    location_change_periods = []
    current_location = None
    current_start_time = None
    current_address = None
    for record in filtered_records:
        location = (record['latitude'], record['longitude'])
        # get the address of the location
        address = get_address_from_coordinates(record['latitude'], record['longitude'])
        if current_location is None:
            current_location = location
            current_start_time = record['timestamp']
            current_address = address
        
        # check if location is at least 100 meters away from the current location
        if (current_location is not None) and (get_distance(current_location, location) > 100) and (address != current_address):
            # If the location has changed, save the current period
            location_change_periods.append({
                'start_time': current_start_time,
                'end_time': record['timestamp'],
                'address': address
            })
            # Update the current location and start time
            current_location = location
            current_start_time = record['timestamp']
            current_address = address
    # Add the last period if it exists
    if current_location is not None and (get_distance(current_location, location) > 100) and (address != current_address):
        location_change_periods.append({
            'start_time': current_start_time,
            'end_time': filtered_records[-1]['timestamp'],
            'address': get_address_from_coordinates(current_location[0], current_location[1])
        })
    return location_change_periods

if __name__ == "__main__":
    uid = "pilot2"
    start_time = "2025-02-19 10:58:00"
    end_time = "2025-02-19 11:24:00"
    # print("Fetching Android location records...")
    # Example usage of the functions
    records = get_android_location_records(uid, start_time, end_time)
    print("Location Records:", records)

    # address = get_address_from_coordinates(37.7749, -122.4194)
    # print("Address from Coordinates:", address)

    # paths = get_location_paths(uid, start_time, end_time)
    # print("Location Paths:", paths)

    # most_frequent_address = get_address_at_timestamp(uid, start_time, end_time)
    # print("Most Frequent Address:", most_frequent_address)

    # get_location_change_periods = get_location_records(uid, start_time, end_time)
    # print("Location Change Periods:", get_location_change_periods)