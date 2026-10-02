import textwrap
from correct_labels_check import *
from data_streams.pixel_steps_data import detect_step_periods_within_time_range
from data_streams.android_phone_usage import get_phone_usage_records
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def retrieving_data(uid, start_time, end_time, a_type="gpt4-gloss"):
    """
    Start time and end time should be in the format "YYYY-MM-DD HH:MM"
    """
    # gt_df = read_annotated_file(uid)
    result_df = get_all_results_from_participants(uid, a_type=a_type)
    original_df = read_annotated_file(uid)

    result_df = parse_results_column(result_df, 'result')
    try:
        # gt_df = clean_activity_labels(gt_df, 'activity')
        result_df = clean_activity_labels(result_df, 'activity')
        original_df = clean_activity_labels(original_df, 'activity')
    except Exception as e:
        # print the traceback
        return None
    
    # gt_df = split_into_minute_level(gt_df)
    result_df = split_into_minute_level(result_df)
    original_df = split_into_minute_level(original_df)

    # if there are duplicated timestamps, append all the activities together, separated by comma
    result_df = result_df.groupby('timestamp').agg({'activity': lambda x: ', '.join(x)}).reset_index()
    original_df = original_df.groupby('timestamp').agg({'activity': lambda x: ', '.join(x)}).reset_index()

    # for each timestamp, check if there are duplicated activities, if so, keep only one
    result_df['activity'] = result_df['activity'].apply(lambda x: ', '.join(sorted(set([i.strip() for i in x.split(',')]))))
    original_df['activity'] = original_df['activity'].apply(lambda x: ', '.join(sorted(set([i.strip() for i in x.split(',')]))))

    # filter by timestamp between start_time and end_time
    # first convert start_time and end_time to datetime objects, and the timestamp column to datetime objects
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    # gt_df['timestamp'] = pd.to_datetime(gt_df['timestamp'], format='HH:MM')
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], format='%H:%M')
    original_df['timestamp'] = pd.to_datetime(original_df['timestamp'], format='%H:%M')
    # set the date to the same date as start_time and end_time
    # gt_df['timestamp'] = gt_df['timestamp'].apply(lambda x: x.replace(year=start_time.year, month=start_time.month, day=start_time.day))
    result_df['timestamp'] = result_df['timestamp'].apply(lambda x: x.replace(year=start_time.year, month=start_time.month, day=start_time.day))
    original_df['timestamp'] = original_df['timestamp'].apply(lambda x: x.replace(year=start_time.year, month=start_time.month, day=start_time.day))
    # filter the dataframe to only include rows between start_time and end_time
    # gt_df = gt_df[(gt_df['timestamp'] >= start_time) & (gt_df['timestamp'] <= end_time)]
    result_df = result_df[(result_df['timestamp'] >= start_time) & (result_df['timestamp'] <= end_time)]
    original_df = original_df[(original_df['timestamp'] >= start_time) & (original_df['timestamp'] <= end_time)]
    # merge the result and original dataframes on timestamp
    merged_df = pd.merge(result_df, original_df, on='timestamp', suffixes=('_result', '_original'))

    # replace upright to standing in both activity columns
    merged_df['activity_result'] = merged_df['activity_result'].replace('upright', 'standing')
    merged_df['activity_original'] = merged_df['activity_original'].replace('upright', 'standing')
    return merged_df

def plot_step_example(uid, start_time, end_time, anno_time=None, anno_text=None, a_type="gpt4-gloss"):
    merged_df = retrieving_data(uid, start_time, end_time, a_type=a_type)
    # print out duplicated timestamps
    step_data = get_steps_records(uid, start_time, end_time)
    phone_df = get_phone_usage_records(uid, start_time, end_time)
    # hr_data = get_garmin_hr_records(uid, f"{participants[uid]} 21:00:00", f"{participants[uid]} 21:13:00")
    # hr_data_2 = get_garmin_hr_records(uid, f"{participants[uid]} 20:53:00", f"{participants[uid]} 20:58:00")

    ### HANDLE STEP DATA ###
    # convert into df
    step_data = pd.DataFrame(step_data)
    # convert the timestamp column to datetime YYYY-MM-DD HH:MM:SS
    step_data['timestamp'] = pd.to_datetime(step_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    # set all the seconds to 0
    step_data['timestamp'] = step_data['timestamp'].dt.floor('min')

    ## HANDLE PHONE USAGE DATA ##
    # convert phone usage into df
    phone_df = pd.DataFrame(phone_df)
    try:
        # convert the timestamp column to datetime YYYY-MM-DD HH:MM:SS
        phone_df['timestamp'] = pd.to_datetime(phone_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        # set all the seconds to 0
        phone_df['timestamp'] = phone_df['timestamp'].dt.floor('min')
    except Exception as e:
        phone_df = pd.DataFrame(columns=['timestamp', 'in_use'])

    ### HANDING HR DATA ### (similar to step data)
    # hr_data = pd.DataFrame(hr_data)
    # hr_data['timestamp'] = pd.to_datetime(hr_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    # # increase all timestamp  by 2 minutes to align with step data delay
    # hr_data['timestamp'] = hr_data['timestamp'] + pd.Timedelta(minutes=2)
    # hr_data_2 = pd.DataFrame(hr_data_2)
    # hr_data_2['timestamp'] = pd.to_datetime(hr_data_2['timestamp'],
    #                                             format='%Y-%m-%d %H:%M:%S')
    # hr_data_2['timestamp'] = hr_data_2['timestamp'] + pd.Timedelta(minutes=7)
    # # append hr_data_2 to hr_data
    # hr_data = pd.concat([hr_data_2, hr_data], ignore_index=True)

    # merge with merged_df on timestamp (keep all rows from merged_df)
    merged_df = pd.merge(merged_df, step_data, on='timestamp', how='left')
    # fill na in steps with 0
    merged_df['steps'] = merged_df['steps'].fillna(0)
    # merge with phone_df on timestamp (keep all rows from merged_df)
    merged_df = pd.merge(merged_df, phone_df[['timestamp', 'in_use']], on='timestamp', how='left')
    # fill na in is_screen_on with False
    merged_df['in_use'] = merged_df['in_use'].fillna(False)
    # if in_use == true, set to "in use", else ""
    merged_df['in_use'] = merged_df['in_use'].apply(lambda x: "in use" if x else "")

    # replace upright to standing in both activity columns
    merged_df['activity_result'] = merged_df['activity_result'].replace('upright', 'standing')
    merged_df['activity_original'] = merged_df['activity_original'].replace('upright', 'standing')

    time_col="timestamp"
    steps_col="steps"
    result_col="activity_result"
    original_col="activity_original"
    df = merged_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Collapse consecutive identical activity into intervals
    def get_intervals(series, timestamps):
        intervals = []
        start = 0
        for i in range(1, len(series)):
            if series[i] != series[i-1]:
                intervals.append((timestamps[start], timestamps[i-1], series[i-1]))
                start = i
        intervals.append((timestamps[start], timestamps.iloc[-1], series.iloc[-1]))
        return intervals

    result_intervals = get_intervals(df[result_col], df[time_col])
    original_intervals = get_intervals(df[original_col], df[time_col])
    try:
        phone_usage_intervals = get_intervals(df['in_use'], df[time_col])
    except Exception as e:
        phone_usage_intervals = []

    # Build color map
    activities = set(df[result_col]) | set(df[original_col])
    palette_list = sns.light_palette("#305BAB", n_colors=4, reverse=False)
    unique_activities = sorted(activities)
    color_map = {act: palette_list[0] for act in unique_activities}
    # print the value of the first color in hex
    color_map["in use"] = palette_list[3]
    color_map[""] = palette_list[0]  # for no phone usage

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 5),
                             sharex=True, gridspec_kw={'height_ratios':[4, 0.5, 1, 1]})
    # fig, axes = plt.subplots(3, 1, figsize=(14, 5),
    #                          sharex=True, gridspec_kw={'height_ratios':[4, 1, 1]})
    # 1. Step bar plot
    axes[0].bar(
        df[time_col],
        df[steps_col],
        width=pd.Timedelta(minutes=1),
        color=palette_list[2],
        edgecolor="black",
        align="center",
        label='Steps (from Pixel Watch)'  # ← Add this line
    )
    axes[0].set_ylabel("Steps count (per min)", fontsize=14)

    # plot hr in the same plot with a line plot, and axis on the right
    # ax2 = axes[0].twinx()
    # ax2.plot(
    #     hr_data['timestamp'],
    #     hr_data['heart_rate'],
    #     color="#cfa454",
    #     marker='o',
    #     markersize=4,
    #     label='Heart Rate (from Pixel Watch)',
    #     alpha=0.7
    # )
    # ax2.set_ylabel("Heart Rate (BPM)", fontsize=14, color='black')
    # ax2.tick_params(axis='y', labelcolor='black')

    # joint legend for both axes
    # lines_1, labels_1 = axes[0].get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # axes[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

    def wrap_text(text, bar_start, bar_end, ax, font_size=16, pad=0.9):
        # Convert bar width to “minutes”
        bar_width_minutes = (mdates.date2num(bar_end) - mdates.date2num(bar_start)) * 24 * 60
        if bar_width_minutes <= 0:
            return text

        # Estimate max characters per line
        chars_per_minute = font_size * 0.3
        max_chars = max(1, int(bar_width_minutes * chars_per_minute * pad))

        # Split text into "comma groups"
        parts = [p.strip() for p in text.split(',')]
        # Re-attach commas except for the last part
        parts = [p + ',' if i < len(parts)-1 else p for i, p in enumerate(parts)]
        # for each parts, split by spaces into small parts, and add all parts into a new list
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(' '))
        parts = new_parts

        # Build lines, each part stays together
        lines = []
        current_line = ""
        for part in parts:
            if len(current_line) + len(part) + 1 <= max_chars:
                # Add to current line
                if current_line:
                    current_line += " " + part
                else:
                    current_line = part
            else:
                # Start new line with this part
                if current_line:
                    lines.append(current_line.strip())
                current_line = part
        if current_line:
            lines.append(current_line.strip())

        return "\n".join(lines)


    def get_contrast_text_color(rgb_color):
        r, g, b = mcolors.to_rgb(rgb_color)
        r, g, b = [x * 255 for x in (r, g, b)]
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return "black" if luminance > 186 else "white"

    def plot_timeline(ax, intervals, row_label):
        if intervals[-1][2] and "riding train" in intervals[-1][2]:
            intervals = intervals[:-1]
        for start, end, activity in intervals:
            # if start and end are less than 2 minutes, skip
            if (end - start) < pd.Timedelta(minutes=1):
                continue
            color = color_map.get(activity, "gray")
            ax.barh(
                row_label, end - start + pd.Timedelta(minutes=1), left=start, height=0.4,
                color=color, edgecolor="black"
            )
            text_color = get_contrast_text_color(color)
            if activity != "in use":
                ax.text(    
                    start + (end - start)/2 + pd.Timedelta(minutes=0.5), 
                    row_label, 
                    wrap_text(activity, start, end, ax, font_size=16),
                    ha="center", va="center", fontsize=16, color=text_color
                )

    # plot phone usage
    plot_timeline(axes[1], phone_usage_intervals, "Phone\nIn Use")
    # 2. Activity Result
    plot_timeline(axes[3], result_intervals, "Fixed\nAnnotations")
    # 3. Activity Original
    plot_timeline(axes[2], original_intervals, "Original\nAnnotations")

    # Axis formatting
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, axis="x", linestyle="--", alpha=0.6)
        ax.tick_params(axis='x', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=16)

    plt.subplots_adjust(hspace=0.3)

    # Add annotation bubble if requested
    if anno_time is not None and anno_text is not None:
        # Ensure anno_time is datetime
        anno_time = pd.to_datetime(anno_time)
        
        # Wrap text
        def wrap_annotation(text, max_chars=50):
            return "\n".join(textwrap.wrap(text, max_chars))
        
        wrapped_text = wrap_annotation(anno_text, max_chars=40)
        
        # Top-left position for annotation text
        x_min, x_max = axes[0].get_xlim()
        y_min, y_max = axes[0].get_ylim()
        x_text = x_min + (x_max - x_min) * 0.01  # 1% from left
        y_text = y_max * 0.95  # near top
        
        # Arrow points to the step value at anno_time
        y_arrow = df.loc[df[time_col] == anno_time, steps_col].values
        if len(y_arrow) > 0:
            y_arrow = y_arrow[0]
        else:
            y_arrow = 10

        axes[0].annotate(
            wrapped_text,
            xy=(anno_time, y_arrow),  # arrow tip
            xycoords="data",
            xytext=(anno_time, y_text),  # text position
            textcoords="data",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=1.5),
            fontsize=14,
            ha="left",
            va="top"
        )

    plt.tight_layout()
    plt.savefig(f"/mnt/study/ari_work/llm-sensemaking/figs/{uid}_activity_timeline.pdf", dpi=300)


if __name__ == "__main__":
    uid = "pilot5"

    participants = {
        'pilot2': '2025-02-20',
        'pilot5': '2025-02-25',
        'pilot6': '2025-02-25',
        'pilot7': '2025-02-27',
        'pilot8': '2025-02-27',
        'pilot9': '2025-03-02',
        'pilot10': '2025-03-10',
        'pilot11': '2025-03-03',
    }

    lst_of_test = [
        # {
        #     "uid": "pilot9",
        #     "start": f"{participants['pilot9']} 12:05:00",
        #     "end": f"{participants['pilot9']} 12:22:00",
        #     "anno_time": f"{participants['pilot9']} 12:12:30",
        #     "anno_text": "[GLOSS]: Step count data shows a brief increase in steps, indicating a short period of walking."
        # },
        {
            "uid": "pilot5",
            "start": f"{participants['pilot5']} 21:00:00",
            "end": f"{participants['pilot5']} 21:15:00",
            "anno_time": f"{participants['pilot5']} 21:05:00",
            "anno_text": "[GLOSS]: Increased step count and elevated heart rate suggest walking activity during this period."
        },
        # {
        #     "uid": "pilot10",
        #     "start": f"{participants['pilot10']} 16:00:00",
        #     "end": f"{participants['pilot10']} 16:25:00",
        #     "anno_time": f"{participants['pilot10']} 16:25:00",
        #     "anno_text": "[GLOSS]: Increased step count and elevated heart rate suggest walking activity during this period."
        # },
        {
            "uid": "pilot11",
            "start": f"{participants['pilot11']} 11:40:00",
            "end": f"{participants['pilot11']} 11:55:00",
            "anno_time": f"{participants['pilot11']} 11:50:00",
            "anno_text": "[GLOSS]: Ambient noise “Telephone, Ringtone” and phone‑usage logs indicate phone use while cooking."
        },
    ]


    start_time = f"{participants[uid]} 21:02:00"
    end_time = f"{participants[uid]} 21:15:00"

    anno_time = f"{participants[uid]} 21:05:30"
    anno_text = "[GLOSS]: Increased step count and elevated heart rate suggest walking activity during this period."
    # df = retrieving_data(uid, start_time, end_time)
    # plot_step_example(uid, start_time, end_time, anno_time, anno_text)
    a_type="gpt4-narrative"
    for test in lst_of_test:
        print(f"Plotting for {test['uid']} from {test['start']} to {test['end']}")
        if isinstance(test, dict):
            plot_step_example(test["uid"], test["start"], test["end"], test["anno_time"], test["anno_text"], a_type=a_type)
        else:
            plot_step_example(test, f"{participants[test]} {start_time}", f"{participants[test]} {end_time}", f"{participants[test]} {anno_time}", anno_text=anno_text, a_type=a_type)