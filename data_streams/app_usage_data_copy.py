import math
import sys
import os
import matplotlib.dates as mdates
from datetime import datetime

from data_processing.plotting_utils import plot_blocks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))
import matplotlib.pyplot as plt


from datetime import datetime
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.lock_unlock_data import get_lock_unlock_blocks

from data_streams.constants import APP_USAGE_LOGS, IOS_LOCK_UNLOCK


def process_records(app_records):
    records = []
    for r in app_records:
        d = {}
        time = datetime.fromtimestamp(r['timestamp'])
        d['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        d['appName'] = r['appName']
        d['status'] = r['status']
        records.append(d)
    return records

def get_app_usage_records(uid, start_time, end_time):
    if (not isinstance(start_time, float)):
        if (isinstance(start_time, str)):
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        start_time = start_time.timestamp()
        end_time = end_time.timestamp()

    app_usage_records = fetch_documents_between_timestamps(uid, start_time, end_time, APP_USAGE_LOGS)
    start_time_ = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    end_time_ = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    lock_unlock_blocks = get_lock_unlock_blocks(uid, start_time_, end_time_)

    if not app_usage_records:
        return []

    app_usage_index = 1
    additional_records = []

    for p in process_records(app_usage_records):
        print(p)

    for l in (lock_unlock_blocks):
        print(l)


    block_index = 0
    for j in range(0, len(app_usage_records)):

        for i in range(block_index, len(lock_unlock_blocks)):
            block_start_time = datetime.strptime(lock_unlock_blocks[i]['start_time'], "%Y-%m-%d %H:%M:%S").timestamp()
            block_end_time = datetime.strptime(lock_unlock_blocks[i]['end_time'], "%Y-%m-%d %H:%M:%S").timestamp()

            if (app_usage_records[j]['timestamp'] > block_end_time):
                continue

        # if(app_usage_records[j]['timestamp'] > block_end_time):
        #     if app_usage_records[j-1]['status'] == "open":
        #         # if you close phone and not close app it closes after few nano secs
        #         if(app_usage_records[j]['status'] == "close" and abs(app_usage_records[j]['timestamp'] - block_end_time) < 4):
        #             app_usage_index = j + 1
        #         else:
        #             additional_records.append({"appName": app_usage_records[j-1]['appName'], "timestamp": block_end_time, "status": "close"})
        #             app_usage_index = j
        #     break

        # if math.floor(app_usage_records[j]['timestamp']) > block_start_time and math.floor(app_usage_records[j]['timestamp']) < block_end_time:
        #     if(lock_unlock_blocks[i]['state'] == "locked"):
        #         print("detected app activity in locked state")
        #         print(app_usage_records[j])
        #         print(block_start_time, block_end_time)
        #         print(lock_unlock_blocks[i])

        if(app_usage_records[j]['appName']['status'] == app_usage_records[j+1]['status']):
            if app_usage_records[j]['status'] == "close":

                    additional_records.append({"appName": app_usage_records[j]['appName'], "timestamp": app_usage_records[j-1]['timestamp'], "status": "open"})
            if app_usage_records[j]['status'] == "open":
                if (app_usage_records[j]['appName']['status'] == app_usage_records[j - 1]['status']):
                    additional_records.append({"appName": app_usage_records[j-1]['appName'], "timestamp": app_usage_records[j]['timestamp'], "status": "close"})
        else:
            if(app_usage_records[j]['status'] == "close" and app_usage_records[j-1]['status'] == "open"):

            if(block_end_time > app_usage_records[j]['timestamp']):
                additional_records.append({"appName": app_usage_records[j-1]['appName'], "timestamp": end_time, "status": "close"})


    app_usage_records.extend(additional_records)

    # print("adding fixed records:", len(additional_records))

    app_usage_records.sort(key=lambda x: x['timestamp'])

    return process_records(app_usage_records)


from datetime import datetime


def get_app_usage_blocks(uid, start_time, end_time):
    app_usage_records = get_app_usage_records(uid, start_time, end_time)

    for r in app_usage_records:
        print(r)

    if not app_usage_records:
        return []
    app_usage_blocks = []

    for i in range(1, len(app_usage_records)):
        # Convert timestamps to datetime objects
        current_timestamp = datetime.strptime(app_usage_records[i]['timestamp'], "%Y-%m-%d %H:%M:%S")
        previous_timestamp = datetime.strptime(app_usage_records[i - 1]['timestamp'], "%Y-%m-%d %H:%M:%S")

        if app_usage_records[i]['appName'] != app_usage_records[i - 1]['appName']:
            if app_usage_records[i - 1]['status'] == "open":
                app_usage_blocks.append({
                    "app": app_usage_records[i - 1]['appName'],
                    "open": app_usage_records[i - 1]['timestamp'],
                    "close": app_usage_records[i]['timestamp'],
                    "duration": (current_timestamp - previous_timestamp).total_seconds()
                })
        else:
            if(not(app_usage_records[i-1]['status'] == "close" and app_usage_records[i]['status'] == "open")):
                app_usage_blocks.append({
                    "app": app_usage_records[i - 1]['appName'],
                    "open": app_usage_records[i - 1]['timestamp'],
                    "close": app_usage_records[i]['timestamp'],
                    "duration": (current_timestamp - previous_timestamp).total_seconds()
                })

    return app_usage_blocks




def plot_app_durations(data):
    # Convert timestamps to datetime and prepare durations and app names
    times = [datetime.strptime(entry['open'], "%Y-%m-%d %H:%M:%S") for entry in data]
    durations = [entry['duration'] for entry in data]
    apps = [entry['app'] for entry in data]

    # Assign a unique color to each app
    unique_apps = set(apps)
    colors = plt.cm.get_cmap('tab10', len(unique_apps))
    app_colors = {app: colors(i) for i, app in enumerate(unique_apps)}

    # Plot each app's durations
    plt.figure(figsize=(12, 6))
    for i in range(len(data)):
        plt.bar(times[i], durations[i], color=app_colors[apps[i]], width=0.001)

    # Formatting the plot
    plt.xlabel('Time')
    plt.ylabel('Duration (seconds)')
    plt.title('App Usage Duration Over Time')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.legend(app_colors.keys(), title="Apps")

    plt.show()


if __name__ == "__main__":
    start_datetime = datetime(2024, 7, 7, 0, 0, 0)
    end_datetime = datetime(2024, 7, 7, 2, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    # app_usage_records = get_app_usage_records('test004', start_timestamp, end_timestamp)
    # print(app_usage_records)
    blocks = get_app_usage_blocks('test004', start_timestamp, end_timestamp)
    print(blocks)
    plot_app_durations(blocks)

    # app_usage_blocks = get_app_usage_blocks(app_usage_records)
    # print(app_usage_blocks)
    # print(generate_app_usage_summary(app_usage_blocks))
    # plot_app_usage(app_usage_blocks)