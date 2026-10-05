import os


def _env_flag(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


# LLM backend: "gpt-4o" (OpenAI), "gpt-5" (OpenAI) or "gpt-oss" (Ollama endpoint in LLM_API_URL)
MODEL = os.getenv("GLOSS4HAR_MODEL", "gpt-oss")
if MODEL not in ("gpt-4o", "gpt-5", "gpt-oss"):
    raise ValueError(f"GLOSS4HAR_MODEL must be one of gpt-4o, gpt-5, gpt-oss (got '{MODEL}')")
USE_AZURE = False
USE_GPT5 = MODEL == "gpt-5"
USE_GPT_OSS = MODEL == "gpt-oss"

# Ablations
ABLATION_PRESENTATION_AGENT = _env_flag("GLOSS4HAR_NO_PRESENTATION")
ABLATION_MEMORY = _env_flag("GLOSS4HAR_NO_MEMORY")

# Give the agents access to the uEMA self-reports (the uEMA condition of task 2)
USE_UEMA = _env_flag("GLOSS4HAR_USE_UEMA")
result_expainations = {
    "get_location_records": """
    This function retrieves a trace of all GPS location records per minutre for a specific user within a given time range. It returns list of dict: A list of GPS location records where each record is represented as a dictionary. Each dictionary contains:
                        - 'timestamp' (int): The timestamp of the GPS record.
                        - 'accuracy' (float): The accuracy of the GPS record in meters.""",

    "get_location_statistical_metrics": """A dictionary containing various location metrics:
                        - "time_spent_at_home" (float): Time spent at home in minutes.
                        - "total_time_all_centers" (float): Total time spent at significant locations in minutes.
                        - "max_displacement" (float): Maximum distance between significant locations in kilometers.
                        - "distance_sum" (float): Total distance traveled between significant locations in kilometers.
                        - "num_loc_visited" (int): Number of distinct significant locations visited.
                        - "displacement_sum" (float): Sum of distances between successive significant locations in kilometers.
                        - "radius_of_gyration" (float): Radius of gyration around the center of mass of significant locations.
                        - "location_entropy" (float): Entropy of the distribution of time spent at significant locations.
                        - "nomalized_location_entropy" (float): Normalized entropy relative to the number of significant locations.
                        - "max_displacement_from_home" (float): Maximum displacement from home in kilometers.""",

    "get_location_paths": """This function extracts and organizes distinct paths taken by a user based on GPS location data within a specified time range. It returns a list of dict: A list of dictionaries where each dictionary represents a path segment. Each dictionary contains:  
                          - 'starting_point' (dict): The starting GPS record of the path, including:
                                - 'latitude' (float): Latitude of the GPS location.
                                - 'longitude' (float): Longitude of the GPS location.
                                - 'timestamp' (int): Timestamp of the GPS record.
                            - 'end_point' (dict): The ending GPS record of the path, including:
                                - 'latitude' (float): Latitude of the GPS location.
                                - 'longitude' (float): Longitude of the GPS location.
                                - 'timestamp' (int): Timestamp of the GPS record.""",

    "get_activity_records": "Retrieves activity records for a given user within a specified time range. Returns a list of activity records, where each record is a dictionary containing the user's UID, timestamp, activity types (e.g., 'cycling', 'walking'), and confidence level of the detection.",
    "get_activity_blocks": "Retrieves contiguous blocks of activity for a given user within a specified time range. Returns a list of dictionaries, each representing an activity block with keys for the activity type (e.g., 'stationary', 'automotive'), start time, and end time of the block.",
    "generate_total_activity": "Calculates the total time spent on each activity by a user within a specified time range. Returns a dictionary where the keys are activity names (e.g., 'Walking', 'Running') and the values are the total time spent on each activity in minutes, calculated by summing up the durations of all occurrences of that activity.",

    "get_address_from_coordinates": "Retrieves the address corresponding to a given latitude and longitude using a precise geocoding service. Returns a string representing the address.",

    "get_phone_steps_stats": "A dictionary containing the total steps, total distance, total floors ascended, and total floors descended within the specified time range.",
    "get_garmin_hr": "A list of heart rate every 30 seconds between the specified timestamps for the given user.",
    "get_hr_summary": "The combined heart rate summary for the specified time range based on instructions.",
    "get_lock_unlock_blocks": "Returns periods of consistent lock state for a user within a specified time range. Retrieves lock/unlock records for a user (`uid`) between `start_time` and `end_time`, and generates a list of time blocks where the lock state remains unchanged.",
    "get_total_lock_unlock_duration": "Summarizes total time spent in each lock state for a user within a specified time range. Retrieves lock/unlock blocks for a user (`uid`) between `start_time` and `end_time`, and calculates the total duration spent in the 'locked' and 'unlocked' states.",
    "get_lock_unlock_summary": "Retrieves a summary of lock unlock phone events for a specific user within a given time range based on instructions provided.",
    "get_location_summary": "Returns a summary of location coordinates based on the provided instructions.",
    "get_activity_summary": "Retrieves a summary of activity records for a specific user within a given time range based on instructions provided.",
    "get_phone_steps_summary": "Retrieves a summary of steps walked, floors ascended, floors descended, distance covered for a specific user within a given time range using steps data from phone based on instructions provided.",
    "get_garmin_steps_summary": "Retrieves a summary of steps walked for a specific user within a given time range using steps data from garmin smart watch based on instructions provided.",
    "get_total_garmin_steps": "Retrieves per totals step records for a specified user within a given time range gathered from garmin smart watch.",
    "get_wifi_blocks": "Retrieves Wi-Fi connection blocks containing status and name of wifi for a user within a specified time range.",
    "get_wifi_usage_summary": "Retrieves a summary of wifi data for a specific user within a given time range based on instructions provided.",
    "generate_wifi_total_duration": "Calculates the total time spent connected to specific Wi-Fi networks or being disconnected by a user within a specified time range.",
    "get_activity_at_given_time": "Retrieves the activity record closest to a given timestamp for a specific user. Returns a dictionary containing the activity type and timestamp of the closest activity record.",
    "get_location_at_given_time": "Retrieves the GPS location record closest to a given timestamp for a specific user. Returns a dictionary containing the closest GPS location record and its timestamp.",
    "get_lock_unlock_state_at_given_time": "Retrieves the lock/unlock state for a given timestamp for a specific user. Returns a dictionary containing the lock state and timestamp of the lock/unlock record.",
    "get_app_usage_blocks": "Retrieves app usage blocks containing app name, open time, close time and duration for a user within a specified time range.",
    "get_most_recent_app": "Retrieves the most recently used app by a user given a time.",
    "get_app_usage_summary": "Retrieves a summary of app usage data for a specific user within a given time range based on instructions provided.",
    "get_phone_steps_through_data_computation": "Generates a Python code snippet to answer a user query using phone steps data.",
}

databases = {
    # "activity database": {
    #     "info": "Contains activity data (e.g., 'stationary,' 'automotive,' 'cycling,' 'walking,' 'running') recorded via accelerometer and gyroscope sensors in the phone. If the phone is not carried, the user is assumed inactive.",
    #     "device": "Phone"
    # },
    # "location database": {
    #     "info": "Contains latitude, longitude, and altitude data per minute, along with functions to fetch address and calculate metrics like time spent at a location and location paths.",
    #     "device": "Phone",
    #     "additional_instructions": "The location database provides functions to calculate physical address but only call it when needed as it is computationally expensive. Do all calculation in latitude and longitude values and call this function only when you need to show the address to the user."
    # },
    # "phone steps database": {
    #     "info": "Contains steps walked, floors ascended, floors descended, and distance covered between two intervals, calculated via the phone.",
    #     "device": "Phone",
    # "additional_instructions": "1) mention that you are using Phone to get step count data. 2) contains steps counts which migh overalap with garmin steps database"
    # },
    # "garmin steps database": {
    #     "info": "Contains steps walked between two intervals, calculated via Garmin smartwatch. Measures the same thing as phone steps database but using Garmin smartwatch.",
    #     "device": "Garmin Smartwatch",
    #     "additional_instructions": "1) mention that you are using Garmin to get step count data. 2) contains steps counts which migh overalap with garmin steps database"
    # },
    # "garmin hr database": {
    #     "info": "Contains heart rate data (per 30 seconds) and functions for summarizing heart rate data recorded from the Garmin smartwatch.",
    #     "device": "Garmin Smartwatch"

    # },
    # "lock unlock database": {
    #     "info": "Contains records of phone lock and unlock times, with functions to extract this information.",
    #     "device": "Phone"
    # },
    # "wifi database": {
    #     "info": "Contains data whether phone is connected to wifi or not. It also contains the wifi name to which phone is connected.",
    #     "device": "Phone"
    # },
    # "app usage database": {
    #     "info": "Contains app usage data, including app names, open and close times, and durations.",
    #     "device": "Phone"
    # },
    # "phone battery database": {
    #     "info": "Contains battery data, including battery left percentage and charging/discharging events.",
    #     "device": "Phone"
    # },
    # "call log database": {
    #     "info": "Contains call log records, including timestamps, call types (e.g., 'incoming,' 'outgoing'), call durations, and phone ringing durations.",
    #     "device": "Phone"
    # },
    # "garmin stress database": {
    #     "info": "Contains physiological stress predictions from ibi data recorded from the Garmin smartwatch. physiological stress might not always be same as psychological stress. The predictions are stress probabilities with near to 1 being more stressed.",
    #     "device": "Garmin Smartwatch"
    # }
    "location database": {
        "info": "Contains GPS location data (latitude, longitude, altitude) recorded via the phone.",
        "device": "Phone",
        "additional_instructions": "The location database can be used to detected activity related to the location, such as home, work, entertainment, etc. It can also detected speed to identify activity like riding train, bus, cycling, ... The location database provides functions to calculate physical address but only call it when needed as it is computationally expensive. Do all calculation in latitude and longitude values and call this function only when you need to show the address to the user."
    },
    "uEMA database": {
        "info": "Contains user's self-report of in-the-moment activity on the watch. Some of the self-reports are voice-based, so the responses in the database are transcribed text.",
        "device": "Watch",
        "additional_instructions": "Some of the transcribed responses might not be accurate (e.g., sitting might be transcribed to setting). The uEMA database is used to retrieve what activity participant did at that moment, not the start time of the activity. To get the exact start and end time of an activity, first use information in other databases where there is a time period of consistent data, then match it with the uEMA to get the exact activity label."
    },
    "heart rate database": {
        "info": "Contains heart rate data (anytime it changes) recorded from the Pixel smartwatch.",
        "device": "Watch",
        "additional_instructions": "The heart rate database is used to understand the user's physical activity level. Combined with step count it can detect moderate to vigorous activities that doesn't involve a lot of steps."
    },
    "phone usage database": {
        "info": "Contains phone usage data. True means the phone is being used, and false means the phone is not being used at the time collected.",
        "device": "Phone",
        "additional_instructions": "The phone usage database is used to detect 'phone using' activity."
    },
    "ambient noise database": {
        "info": "Contains ambient noise data recorded from the Pixel smartwatch. The data is recorded every 5 minutes.",
        "device": "Watch",
        "additional_instructions": "The noises are detected using Google YAMNet model, which classifies the noise into 521 classes. Some of the predictions might be inaccurate or not related to the user's activity."
    },
    "step count database": {
        "info": "Contains step count data recorded from the Pixel smartwatch. The data is recorded every minute.",
        "device": "Watch",
        "additional_instructions": "The step count database is used to detect activities such as walking, running. Is used with heart rate or location, it can be used to detect activities that does not involve walking or running, such as cycling, riding car, etc.",
    },
    "skin temperature database": {
        "info": "Contains skin temperature data recorded from the Pixel smartwatch. The data is recorded every minute.",
        "device": "Watch",
        "additional_instructions": "The skin temperature database can be used to detect activities that involve changes in skin temperature, such as exercise or sleep.",
    },
    "watch wear database": {
        "info": "Contains data whether the watch is worn or not. True means the watch is worn, and false means the watch is not worn at the time collected.",
        "device": "Watch",
        "additional_instructions": "The watch wear database is used to detect activities such as wearing the watch, taking it off, or putting it on."
    }
    # ,
    # "wrist AUC database": {
    #     "info": "Contains wrist AUC data (calculated via accelerometer data) recorded from the Pixel smartwatch. The data is recorded every 10 seconds.",
    #     "device": "Watch",
    #     "additional_instructions": "Wrist AUC <10 indicates the watch is not worn, <300 indicates low movement, <1000 indicates moderate movement. But this is based on the participants baseline level.",
    # }
}

if not USE_UEMA:
    del databases["uEMA database"]
