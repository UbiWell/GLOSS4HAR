import math
import sys
import os
from datetime import datetime, timedelta

import pandas as pd

from data_processing.plotting_utils import plot_blocks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

from data_processing.data_processing_utils import fetch_documents_between_timestamps

from data_processing.data_processing_utils import fetch_first_and_last_document
from app_usage_data import get_app_usage_records, get_total_app_usage
from data_streams.constants import APP_USAGE_LOGS


def calc_hours_between(d1, d2):
    # Ensure both dates are in datetime format
    if isinstance(d1, str):
        d1 = datetime.strptime(d1, '%Y-%m-%d %H:%M:%S')
    if isinstance(d2, str):
        d2 = datetime.strptime(d2, '%Y-%m-%d %H:%M:%S')

    # Calculate the difference and convert to hours
    delta = d2 - d1
    return delta.total_seconds() / 3600

uid = 'test006'

uids = ['test006']

for u in uids:
    first, last = fetch_first_and_last_document(u, APP_USAGE_LOGS)

    dt_object_first = datetime.fromtimestamp(first['timestamp'])
    formatted_first = dt_object_first.strftime('%Y-%m-%d %H:%M:%S')
    dt_object_last = datetime.fromtimestamp(last['timestamp'])
    formatted_last = dt_object_last.strftime('%Y-%m-%d %H:%M:%S')

    print(f"hours for uid {u}: ", calc_hours_between(dt_object_first, dt_object_last))

app_map = {
    "SNAP": "SnapChat",
    "IG": "Instagram",
    "TT": "TikTok",
    "IM": "iMessage",
    "SAFA": "Safari",
    "APPMU": "App Store/Updates",
    "CAM": "Camera",
    "CALL": "Phone Calls",
    "PHO": "Photos",
    "TWIT": "Twitter",
    "PIN": "Pinterest",
    "SPOT": "Spotify",
    "FT": "FaceTime",
    "GC": "Google Chrome",
    "CAN": "Canva",
    "GM": "Gmail",
    "YT": "YouTube",
    "PS": "Photoshop",
    "RB": "Red Book"
}


apps = {
    'test007': ['imessage', 'APPMU', 'SnapChat', 'IG', 'TT', 'CALL', 'TWIT', "PIN"],
    'test008': ["imessage", "TT", "IG", "CALL", "SPOT", "FT", "GC", "PIN"],
    'test009': ["TT", "imessage", "IG", "CAM", "SPOT", "CAN", "GM", "YT", "CALL", "FT"],
    'test006': ["SnapChat", "Instagram", "TikTok", "iMessage", "Safari", "Apple Music", "Camera", "Phone Calls", "Photos", "Photoshop", "YouTube", "FaceTime", "Red Book"]
}


rows = []
current_time = dt_object_first
while current_time < dt_object_last:
    next_time = current_time + timedelta(hours=2)
    hour_of_day = current_time.hour

    formatted_current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_next_time = next_time.strftime('%Y-%m-%d %H:%M:%S')

    row = [uid, formatted_current_time, formatted_next_time, hour_of_day]
    # Call the function with current time and the next hour as end time
    total_app_usage = get_total_app_usage(uid, formatted_current_time, formatted_next_time)
    if(total_app_usage != {}):
        print(formatted_current_time, formatted_next_time)
    for app in apps[uid]:
        duration = 0
        count = 0
        if app in total_app_usage:
            app_usage = total_app_usage[app]
            duration = app_usage['total_duration']
            count = app_usage['count']

        for app_ in total_app_usage:
            if app_ not in apps[uid]:
                print("missing app", app_)
                sys.exit("missing app: add it to the list of apps")

        row.extend([count, duration])
    # Move to the next hour
    rows.append(row)
    current_time = next_time


df = pd.DataFrame(rows)
cols =[ 'uid', 'start_time', 'end_time', 'hour_of_day']

for app in apps[uid]:
    cols.extend([app + '_count', app + '_duration'])

df.columns = cols

df.to_csv(f'{uid}_app_usage.csv', index=False)

