import random
import sys
import os
import pickle
import langchain_openai as lcai
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import numpy as np
from prompts import *
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))

import sensemaking_process
from data_processing.data_processing_utils import fetch_first_and_last_document, fetch_documents_between_timestamps
from data_streams.constants import IOS_LOCATION, GARMIN_HR, GARMIN_STRESS
from datetime import datetime, timedelta
import pandas as pd
from agents.gpt_utils import invoke_with_retry
from agents.constants import databases


participants = {
    'pilot2': '2025/02/19',
    'pilot8': '2025/02/27',
    'pilot5': '2025/02/25',
    'pilot6': '2025/02/25',
    'pilot7': '2025/02/27',
    'pilot9': '2025/03/02',
    'pilot10': '2025/03/10',
    'pilot11': '2025/03/03',
}

time_chunks = ["08:00-08:15", "08:15-08:30", "08:30-08:45", "08:45-09:00",
               "09:00-09:15", "09:15-09:30", "09:30-09:45", "09:45-10:00",
               "10:00-10:15", "10:15-10:30", "10:30-10:45", "10:45-11:00",
               "11:00-11:15", "11:15-11:30", "11:30-11:45", "11:45-12:00",
               "12:00-12:15", "12:15-12:30", "12:30-12:45", "12:45-13:00",
               "13:00-13:15", "13:15-13:30", "13:30-13:45", "13:45-14:00",
               "14:00-14:15", "14:15-14:30", "14:30-14:45", "14:45-15:00",
               "15:00-15:15", "15:15-15:30", "15:30-15:45", "15:45-16:00",
               "16:00-16:15", "16:15-16:30", "16:30-16:45", "16:45-17:00",  
               "17:00-17:15", "17:15-17:30", "17:30-17:45", "17:45-18:00",
               "18:00-18:15", "18:15-18:30", "18:30-18:45", "18:45-19:00",
               "19:00-19:15", "19:15-19:30", "19:30-19:45", "19:45-20:00",
               "20:00-20:15", "20:15-20:30", "20:30-20:45", "20:45-21:00",
               "21:00-21:15", "21:15-21:30", "21:30-21:45", "21:45-22:00"]


def run_individual_query(subject_id, memory = '', activities = "", uEMA_sim="", timeline_memory="", time_period='9:05-9:10am'):
    query = get_real_time_sim_prompt(subject_id, time_period, participants, uEMA_sim, timeline_memory)
    sensemaker = sensemaking_process.SenseMaker(
                    query,
                    """
                    - Only use the postures among this list: 'sitting', 'standing', 'lying down', 'reclining', 'upright'
                    - Only use the activities among this list: 'video gaming',  'walking', 'stair climbing',  'getting ready',  'driving',  'bicycling',  'vigorous bicycling',  'aerobics',  'cleaning',  'cooking',  'laundry',  'playing with pet',  'listening to music',  'watching movies/TV',  'studying',  'reading',  'riding car',  'riding train',  'riding bus',  'playing musical instruments',  'attending meeting',  'computer using', 'phone using',  'running',  'getting dressed',  'grooming',  'using bathroom',  'eating',  'talking','strength training',  'washing dishes',  'carrying groceries',  'putting away groceries',  'shopping',  'making bed',  'packing/unpacking',  'sleeping',  'playing sports'
                    - Do not use any activity that is not in the list. use closest activity from the list if you are not sure.
                    """
                )
    if memory != '':
        sensemaker.memory = "This is the timeline constructed so far. Update it if necessary and include in final response:\n" + memory
    sensemaker.make_sense()

    return sensemaker, query

def parse_result_answer(answer):
    # split by new line
    lines = answer.split('\n')
    parsed_labels = []
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        # check if line contains : posture: and ; activities: and ; reasoning:
        if ': posture:' in line and '; activities:' in line and '; reasoning:' in line:
            try:
                time_part = line.split(': posture:')[0].strip()
                posture_part = line.split(': posture:')[1].split('; activities:')[0].strip()
                activities_part = line.split('; activities:')[1].split('; reasoning:')[0].strip()
                reasoning_part = line.split('; reasoning:')[1].strip()
                parsed_labels.append({
                    'time': time_part,
                    'posture': posture_part,
                    'activities': activities_part,
                    'reasoning': reasoning_part
                })
            except Exception as e:
                print(f"Error parsing line: {line}, error: {e}")
                continue
    return parsed_labels

def run_sim_query(csv_file):
    # parse the csv link to get the subject_id: /mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3/pilot2_cleaned.csv
    subject = csv_file.split('/')[-1].split('_')[0]
    # create the name of the result file
    cleaned_csv_file = csv_file.replace('.csv', '_sim_gpt4.csv')
    # read the activities from the csv file
    gt_df = pd.read_csv(csv_file)

    # if no file exists, create one 
    if not os.path.exists(cleaned_csv_file):
        cleaned_df = pd.DataFrame()
        # create the columns subject,date,uncertainty_start,uncertainty_end,labels,result,action_plan,memory,understanding,information_requests,function_calls
        cleaned_df['subject'] = ''
        cleaned_df['date'] = ''
        cleaned_df['uncertainty_start'] = ''
        cleaned_df['uncertainty_end'] = ''
        cleaned_df["prompt"] = ''
        cleaned_df['labels'] = ''
        cleaned_df['result'] = ''
        cleaned_df['action_plan'] = ''
        cleaned_df['memory'] = ''
        cleaned_df['understanding'] = ''
        cleaned_df['information_requests'] = ''
        cleaned_df['function_calls'] = ''
        cleaned_df.to_csv(cleaned_csv_file, index=False)
    # read the result file, if exist
    cleaned_df = pd.read_csv(cleaned_csv_file)

    memory_past_annotations = ''

    # if cleaned_df is not empty, get the last answer as memory_past_annotations
    # if not cleaned_df.empty:
    #     last_memory = cleaned_df.iloc[-1]['memory']
    #     memory_past_annotations = last_memory

    uEMA_sim = ""

    # get the subject, date, labels, uncertainty_start and uncertainty_end
    for hours in time_chunks:
        time_period = hours

        # check if time period already exists in cleaned_df (uncertainty_start-uncertainty_end)
        if ((cleaned_df['subject'] == subject) & (cleaned_df['uncertainty_start'] == time_period.split('-')[0]) & (cleaned_df['uncertainty_end'] == time_period.split('-')[1])).any():
            continue

        try:
            # create the query
            result, query = run_individual_query(
                subject,
                memory=memory_past_annotations,
                time_period=time_period,
                uEMA_sim=uEMA_sim,
                timeline_memory=memory_past_annotations
            )
            new_row = {
                'subject': subject,
                'date': participants[subject],
                'uncertainty_start': time_period.split('-')[0],
                'uncertainty_end': time_period.split('-')[1],
                "prompt": str(query),
                'labels': '',  # no labels for this task
                'result': str(result.answer),
                'action_plan': str(result.hypothesis),
                'memory': str(result.memory),
                'understanding': str(result.understanding),
                'information_requests': str(result.information_request),
                'function_calls': str(result.function_calls)
            }
            cleaned_df = pd.concat([cleaned_df, pd.DataFrame([new_row])], ignore_index=True)
            
            parse_answer = parse_result_answer(result.answer)
            # get the last labels from parse_answer
            predicted_labels = parse_answer[-1]['posture'] + ", " + parse_answer[-1]['activities']
            predicted_labels = predicted_labels.replace(', ', ',').split(',')
            current_time_stamp = hours.split('-')[1]
            try:
            
                # get the actual activities at current_time_stamp from gt_df
                gt_activities = gt_df.loc[gt_df['timestamp'] == current_time_stamp, 'locomotion'].iloc[0]

                gt_activities = gt_activities.split(', ')

                # create a string that compared the predicted and gt activities
                # show the intersection as reported to do, and the difference (only in the predicted) as not reported
                have_done = set(predicted_labels).intersection(gt_activities)
                not_done = set(predicted_labels) - set(gt_activities)

                not_predicted = set(gt_activities) - set(predicted_labels)
                if have_done == set():
                    comparison_str = f"Up until {current_time_stamp} --- Participants did not report doing any of: {predicted_labels}, but have done: {not_predicted}\n"
                else:
                    comparison_str = f"Up until {current_time_stamp} --- Participants reported to have done: {have_done} and not doing: {not_done}. They also did {not_predicted}.\n"
                uEMA_sim += comparison_str
            except Exception as e:
                uEMA_sim +=  f"Participants did not respond at time {current_time_stamp}\n"
                # print the traceback
                traceback.print_exc()
            # update the memory with the result
            memory_past_annotations = f"{new_row['result']}\n"
        except Exception as e:
            print(f"Error processing time period {time_period}: {e}")
        cleaned_df.to_csv(cleaned_csv_file, index=False)
    return cleaned_df

if __name__ == "__main__":
    # run_individual_query('pilot2', activity="walking, standing", time_period="9:05-9:10am")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot2_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot5_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot6_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot7_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot8_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot9_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot10_cleaned.csv")
    run_sim_query("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_sim/pilot11_cleaned.csv")