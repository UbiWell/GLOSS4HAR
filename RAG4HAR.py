"""Vanilla single-LLM baseline for task 1 (annotation correction).

The sensor data for each annotated period is summarized by the data-stream functions and placed directly in one
gpt-4o prompt, without the GLOSS4HAR agents.

    python RAG4HAR.py --annotations annotations/pilot2_cleaned.csv
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from GLOSS4HAR import participants
from data_streams.android_phone_usage import get_phone_usage_period
from data_streams.pixel_ambient_noises import get_pixel_ambient_noise_records
from data_streams.pixel_hr import get_change_point_hr
from data_streams.pixel_steps_data import detect_step_periods_within_time_range
from data_streams.pixel_wear_detection import detect_non_wear_period_within_time_range


def create_vanilla_query(subject_id, annotations, time_period, start_time, stop_time):
    """
    Create a vanilla query for the sensemaking process.
    """

    # get the start and stop time by appending the date to the time
    start_time = f"{participants[subject_id]} {start_time}"
    stop_time = f"{participants[subject_id]} {stop_time}"

    # reformat the start and stop time to the format of YYYY-MM-DD HH:MM:SS
    start_time = datetime.strptime(start_time, "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")
    stop_time = datetime.strptime(stop_time, "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")

    # get the data (location, hr, step, phone usage, ambient noise, skin temp, uEMA, wear detection) for the subject_id
    hr_data = get_change_point_hr(subject_id, start_time, stop_time)
    steps_data = detect_step_periods_within_time_range(subject_id, start_time, stop_time)
    phone_usage_data = get_phone_usage_period(subject_id, start_time, stop_time)
    ambient_noise_data = get_pixel_ambient_noise_records(subject_id, start_time, stop_time)
    # skin_temp_data = get_change_point_skin_temp(subject_id, start_time, stop_time)
    wear_detection_data = detect_non_wear_period_within_time_range(subject_id, start_time, stop_time)

    initial_query = f"""{subject_id} annotated {annotations} during entire {time_period}.\n

            Based on the data for {participants[subject_id]}, provide a more accurate annotations. A person might be doing multiple activities/postures during that time, in that case provide multiple corrected annotations.\n
            Trust the activity they report but correct it when needed. They usually remember the activity right but might mess up the time period.\n

                - A lot of the time participants under or over estimate the start and stop time. Check data 10 minutes before and after so that you can correct it.\n
                - Check the start and stop time of the activity and correct if needed.\n
                - If the activity is not correct, suggest a different activity that they were doing during that time period.\n
                - Only include annotations within {time_period}, unless {annotations} lasted longer.\n
                - If two annotations have the same posture and activity and is consecutive, merge them into one label with the start time of the first label and end time of the second label.\n
                - Do not output labels with overlappping time.
                
            {'-' * 100}\n
            This is the data for {subject_id} from {start_time} to {stop_time}:\n
            - Heart Rate Data (change detection):**\n
            {hr_data}\n
            **Steps Data (periods with steps):**\n
            {steps_data}\n
            **Periods of using phone:**\n
            {phone_usage_data}\n
            **Ambient Noise Data:**\n
            {ambient_noise_data}\n

            **Periods of wearing watch:\n
            {wear_detection_data}\n
            {'-' * 100}\n
            Please provide the corrected annotations for {subject_id} based on the data above. \n\n

            - Only use the postures among this list: 'sitting', 'standing', 'lying down', 'reclining', 'upright'\n
            - Only use the activities among this list: 'video gaming',  'walking', 'stair climbing',  'getting ready',  'driving',  'bicycling',  'vigorous bicycling',  'aerobics',  'cleaning',  'cooking',  'laundry',  'playing with pet',  'listening to music',  'watching movies/TV',  'studying',  'reading',  'riding car',  'riding train',  'riding bus',  'playing musical instruments',  'attending meeting',  'computer using', 'phone using',  'running',  'getting dressed',  'grooming',  'using bathroom',  'eating',  'talking','strength training',  'washing dishes',  'carrying groceries',  'putting away groceries',  'shopping',  'making bed',  'packing/unpacking',  'sleeping',  'playing sports'\n
            - Do not use any activity that is not in the list. use closest activity from the list if you are not sure.\n
            - Do not add seconds in the final time, only use HH:MM.\n
            - Respond the corrected labels in the format of a list of: - [start time]-[end time]: posture: [posture]; activities: [activities]; reasoning: [reasoning].\n
            - Do not change the format or add additional symbol like * or new line.\n

            Example response:
            - 08:55-09:03: posture: upright; activities: carrying groceries; reasoning: Minimal movement and moderate heart rate suggest light activity consistent with carrying items while upright.
            - 09:03-09:18: posture: upright; activities: carrying groceries; reasoning: Ambient noise suggests outdoor activity, aligning with carrying items while upright.
            - 09:18-09:20: posture: upright; activities: putting away groceries; reasoning: Transition to indoor noise and reduced heart rate suggest less active state, likely putting away groceries.
                """
    return initial_query


def run_query(query):
    """
    Run the query through GPT-4o
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    resp = model.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ])

    return resp.content


def run_query_on_annotations(csv_file, output_csv):
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        query = create_vanilla_query(
            row['subject'],
            row['labels'],
            f"{row['uncertainty_start']}-{row['uncertainty_end']}",
            row['uncertainty_start'],
            row['uncertainty_end'],
        )
        df.at[index, 'result'] = run_query(query)
        df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotations', required=True, help='CSV of annotations to correct')
    parser.add_argument('--output', help='results CSV (default: results/vanilla_<annotations name>.csv)')
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.annotations))[0]
    output = args.output or os.path.join('results', f'vanilla_{name}.csv')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    run_query_on_annotations(args.annotations, output)
    print(f"Results written to {output}")
