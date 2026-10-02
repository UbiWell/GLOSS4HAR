import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime, timedelta
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import IOS_BRIGHTNESS, time_zone_dict
import matplotlib.pyplot as plt
from agents.coding_agent import run_coding_agent
import pytz

functions = {
    "BRIGHTNESS1": {
        "name": "get_brightness_records",
        "description": "Retrieves phone brigtness records between (O and 1) for a specified user within a given time range.  O mean no brightness and 1 means high brightness. brightness can be used as proxy for ambient light of the environment.",
        "usecase": ["code_generation", "function_calling"],
        "params": {
            "uid": {"type": "str", "description": "The unique identifier for the user."},
            "start_time": {"type": "str", "description": "The start time of the period."},
            "end_time": {"type": "str", "description": "The end time of the period."}
        },
        "returns": "list containing brightness per minute ",
        "example": "{'timestamp': '2024-07-09 12:04:00', 'brightness': 0.0}"
    },
    "BRIGHTNESS2": {
        "name": "get_brightness_at_time",
        "usecase": ["code_generation", "function_calling"],
        "description": "Retrieves closest record to a given time. phone brigtness records between (O and 1) for a specified user within a given time range between 0 and 1.  O mean no brightness and 1 means high brightness. brightness can be used as proxy for ambient light of the environment",
        "function_call_instructions": "only call this function when difference between startime and endtime are less than 5 hours apart",
        "params": {
            "uid": {"type": "str",
                    "description": "The unique identifier for the user whose steps data is to be fetched."},
            "given_time": {"type": "str",
                           "description": "the given time."},
        },
        "returns": "Closest brightness record.",
        "example": "[{'start_timestamp': '2024-07-20 00:01:00', 'steps_timestamp': '2024-07-20 00:02:00', 'steps': 0.0, 'total_steps': 0.0}, {'start_timestamp': '2024-07-20 00:02:00', 'steps_timestamp': '2024-07-20 00:03:00', 'steps': 0.0, 'total_steps': 0.0}]"
    },
    "BRIGHTNESS3": {
        "name": "get_garmin_steps_through_data_computation",
        "usecase": [],
        "description": "This function can programatically generates a python code and answer the user query using data in the garmin steps database",
        "function_call_instructions": "Call this function to do perform calculation and computation on garmin steps data.",
        "params": {
            "user_query": {"type": "str",
                           "description": "the query user asked."},
        },
        "returns": "Provides answer to the query by performing computations data in the your database"
    },

    "BRIGHTNESS4": {
        "name": "get_garmin_steps_summary",
        "usecase": ["function_calling"],
        "description": "Retrieves a summary of steps recorded using Garmin smart watch for a specific user within a given time range based on instructions provided.",
        "function_call_instructions": "only call this function when other functions can't answer the query. Do not call this function if startime and endtime are more than 24 hours apart",
        "params": {
            "uid": {"type": "str",
                    "description": "The unique identifier for the user whose steps data is to be fetched."},
            "start_time": {"type": "str",
                           "description": "The start of the time range for which steps data is required."},
            "end_time": {"type": "str",
                         "description": "The end of the time range for which steps data is required."},
            "instructions": {"type": "str", "description": "Specific instructions for summarizing the data."}
        },
        "returns": "A summary of steps records based on the provided instructions."
    }
}


def get_brightness_records(uid, start_time, end_time):

    user_timezone = time_zone_dict.get(uid, "UTC")
    timezone = pytz.timezone("America/New_York") if user_timezone == "est" else pytz.timezone(user_timezone)

    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = timezone.localize(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
            end_time = timezone.localize(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")).astimezone(pytz.UTC)
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    brightness_records = fetch_documents_between_timestamps(uid, start_time, end_time, IOS_BRIGHTNESS)
    return process_records(uid, brightness_records)

def get_brightness_at_time(uid, given_time):

    given_time_ = datetime.strptime(given_time, "%Y-%m-%d %H:%M:%S")

    # Define the start and end time with a 5-minute range
    start_time = given_time_ - timedelta(minutes=10)
    end_time = given_time_ + timedelta(minutes=10)

    brightness_records = get_brightness_records(uid, start_time, end_time)

    closest_record = min(brightness_records, key=lambda record: abs(
        datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S") - given_time_))

    return {'brightness': closest_record['brightness'], 'timestamp': closest_record['timestamp']}


def process_records(uid, brightness_records):
    records = []
    uid_timezone = time_zone_dict.get(uid, 'utc')  # Get UID-specific timezone or default to UTC
    timezone = pytz.timezone("America/New_York") if uid_timezone == "est" else pytz.utc
    for r in brightness_records:
        d = {}
        timestamp = datetime.fromtimestamp(r['timestamp'], pytz.utc).astimezone(timezone)
        d['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        d['brightness'] = r['brightness']
        records.append(d)
    return records


def get_brightness_through_data_computation(user_query):
    include_statements = """
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
    
    from datetime import datetime, timedelta
    from data_processing.data_processing_utils import fetch_documents_between_timestamps
    from data_streams.constants import IOS_BRIGHTNESS, time_zone_dict
    import matplotlib.pyplot as plt
    from agents.coding_agent import run_coding_agent
    import pytz"""

    function_imports = "from from data_streams.brightness import function_name"


    results = run_coding_agent(user_query=user_query, database=["phone brightness database"], functions=functions,
                               include_statements=include_statements, function_imports=function_imports)
    if (not results.messages):
        return "The code generation couldn't answer this query. Please try again later."
    else:
        return results.messages[-1].content.replace("TERMINATE", "")

def plot_brightness(brightness_records):
    brightness = [entry['brightness'] for entry in brightness_records]
    timestamps = [datetime.fromtimestamp(entry['timestamp']) for entry in brightness_records]
    print(timestamps)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.step(timestamps, brightness, where='post', marker='o', linestyle='-', color='b')
    plt.xlabel('Time')
    plt.ylabel('Brightness')
    plt.title('Brightness vs Time (Step Plot)')
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    start_datetime = datetime(2024, 7, 9, 12, 0, 0)
    end_datetime = datetime(2024, 7, 9, 14, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    brightness_records = get_brightness_records('test004', start_timestamp, end_timestamp)
    print(brightness_records)