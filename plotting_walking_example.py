import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import os, sys
import ast
from collections import Counter
import textwrap
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MinuteLocator
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))

from data_streams import pixel_steps_data, android_phone_usage
uid = "pilot9"
start_time = "2025-03-02 12:10:00"
end_time = "2025-03-02 13:00:00"

# plot the steps data as bar chart, with x-axis as timestamp and y-axis as steps
step_data = pixel_steps_data.get_steps_records(uid, start_time, end_time)
timestamps = [record['timestamp'] for record in step_data]
# convert timestamps to datetime objects for better plotting (no timezone handling here) timestamp is string of format 'YYYY-MM-DD HH:MM:SS'
timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
step_data.sort(key=lambda x: x['timestamp'])

# convert it back to HH:MM format for plotting
# timestamps = [ts.strftime('%H:%M') for ts in timestamps]
steps = [record['steps'] for record in step_data]

# get the phone usage data
phone_usage_data = android_phone_usage.get_phone_usage_records(uid, start_time, end_time)
# only keep the records where in_use is True
phone_usage_data = [record for record in phone_usage_data if record['in_use'] == 'True']
phone_timestamps = [record['timestamp'] for record in phone_usage_data]
# convert timestamps to datetime objects for better plotting (no timezone handling here) timestamp is string of format 'YYYY-MM-DD HH:MM:SS'
phone_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in phone_timestamps]
phone_usage_data.sort(key=lambda x: x['timestamp']) 
# convert 'in_use' to boolean
for record in phone_usage_data:
    record['in_use'] = True if record['in_use'] == 'True' else False

phone_in_use = [record['in_use'] for record in phone_usage_data]
phone_in_use = [in_use for in_use, ts in zip(phone_in_use, phone_timestamps) if in_use]
phone_timestamps = [ts for in_use, ts in zip(phone_in_use, phone_timestamps) if in_use] 


# read this file: /mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/GLOSS4HAR_results_pilot9_merged.csv
results_file = f'/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/GLOSS4HAR_results_{uid}_merged.csv'
results_df = pd.read_csv(results_file)
results_df['start_time'] = pd.to_datetime(results_df['start_time'])

# only grab the data between start_time and end_time
start_time_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
results_df = results_df[(results_df['start_time'] >= start_time_dt) & (results_df['start_time'] <= end_time_dt)]
# valid_labels = {'walking', 'sitting', 'phone using'}
valid_labels = {'walking', 'sitting', 'riding train'}
# valid_labels = {'walking', 'sitting', 'running', 'computer using', 'standing'}

# Safely convert stringified lists to real lists and filter valid labels
def parse_and_filter(label_list):
    if isinstance(label_list, str):
        label_list = ast.literal_eval(label_list)  # safer than eval
    # replace 'running' with 'walking' if it exists
    label_list = ['walking' if label == 'running' else label for label in label_list]
    # filter out invalid labels
    labels = str([label for label in label_list if label in valid_labels])
    # add a new line character at the back of each comma-separated label
    # labels = labels.replace(',', ',\n')
    return labels


results_df['labels_human'] = results_df['labels_human'].apply(parse_and_filter)
results_df['labels_gloss'] = results_df['labels_gloss'].apply(parse_and_filter)
# reindex the dataframe
results_df = results_df.reset_index(drop=True)
def remove_temporally_unique_labels_inplace(df, columns=['labels_human', 'labels_gloss']):
    for col in columns:
        # loop through the entries in the column
        for i in range(1, len(df) - 1):
            # check if i is the same with i-1 and i+1
            if df[col].iloc[i] == df[col].iloc[i - 1] or df[col].iloc[i] == df[col].iloc[i + 1]:
                continue  # if they are the same, skip
            # if i is different from i-1 and i+1, we check if it is unique
            if df[col].iloc[i] != df[col].iloc[i - 1] and df[col].iloc[i] != df[col].iloc[i + 1]:
                # if it is unique, we set it to the previous value
                df.at[i, col] = df[col].iloc[i - 1]  # or df[col].iloc[i + 1], they are the same in this case

    # remove rows where there is None   
    df = df.dropna(subset=columns)
    return df

results_df = remove_temporally_unique_labels_inplace(results_df)

# explode the labels_human and labels_gloss columns into multiple rows
results_df['labels_human'] = results_df['labels_human'].apply(lambda x: x.strip('[]').split(', '))
results_df['labels_gloss'] = results_df['labels_gloss'].apply(lambda x: x.strip('[]').split(', '))

print(results_df)

# Explode the DataFrame to have one label per row
results_df = results_df.explode('labels_human')
results_df = results_df.explode('labels_gloss')
# Step 1: Convert string timestamps to datetime (if needed)
if isinstance(timestamps[0], str):
    timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]

# 1. Collect unique labels from both columns
results_df_na = results_df[(results_df['labels_gloss'] != '')]

all_labels = sorted(set(results_df['labels_human'].unique()).union(set(results_df_na['labels_gloss'].unique())))
label_to_y = {label: i for i, label in enumerate(all_labels)}
# 2. Map string labels to y values
results_df['y_gloss'] = results_df['labels_gloss'].map(label_to_y) - 0.1
results_df['y_human'] = results_df['labels_human'].map(label_to_y) + 0.1  # offset for visibility

# 3. Create the plot
fig, axs = plt.subplots(
    2, 1,
    figsize=(6, 3.5),
    sharex=True,
    gridspec_kw={'height_ratios': [1.5,3]}  # Adjust ratios as needed
)
# fig, axs = plt.subplots(
#     3, 1,
#     figsize=(6, 3.5),
#     sharex=True,
#     gridspec_kw={'height_ratios': [8,1,11]}  # Adjust ratios as needed
# )

step_ax = axs[0]
label_ax = axs[1]
# phone_ax = axs[2]

# --- Subplot 1: Steps Bar Chart ---
step_ax.bar(timestamps, steps, color='#1F78B4', width=0.001)
step_ax.set_ylabel('Steps count\n(per minute)')
step_ax.yaxis.set_label_coords(-0.2, 0.5)

# Subplot 1.1: Phone Usage Scatterplot (only plot when True)
# if phone_in_use:
#     phone_ax.scatter(phone_timestamps, [1] * len(phone_timestamps), color='#1F78B4', alpha=1, label='Phone in use')
#     phone_ax.set_yticks([1])
#     phone_ax.set_yticklabels(['Phone interactive'])
#     phone_ax.yaxis.set_label_coords(-0.2, 0.5)

# --- Subplot 2: Human and GLOSS labels ---
sc1 = label_ax.scatter(results_df['start_time'], results_df['y_gloss'], color='#708090', alpha=1, label='GLOSS predictions')
sc2 = label_ax.scatter(results_df['start_time'], results_df['y_human'], color='#D5C27D', alpha=1, label='Participant annotations')

# Set y-axis ticks to labels
label_ax.set_yticks(list(label_to_y.values()))
label_ax.set_yticklabels(list(label_to_y.keys()))
label_ax.set_ylabel('')
label_ax.yaxis.set_label_coords(-0.2, 0.5)
label_ax.grid(axis='y')

# --- Format x-axis ---
label_ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
label_ax.xaxis.set_major_locator(MinuteLocator(interval=5))
label_ax.tick_params(axis='x', rotation=45)

# --- Add legend outside the plot (below), with 2 columns ---
fig.legend(
    handles=[sc1, sc2],
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.62, -0.04),
    frameon=True,
    fontsize='small',
    edgecolor='black',     # black border
    fancybox=False         # square corner (optional)
)


plt.tight_layout()
plt.savefig('/mnt/study/ari_work/llm-sensemaking/combined_plot.png', dpi=300, bbox_inches='tight')
plt.show()