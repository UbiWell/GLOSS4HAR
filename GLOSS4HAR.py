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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))

import sensemaking_process
import pandas as pd
from agents.constants import databases, ABLATION_PRESENTATION_AGENT, ABLATION_MEMORY


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

if ABLATION_PRESENTATION_AGENT:
    presentation_format = "- include both posture and activity in your responses"
else:
    presentation_format = """
                            - Only use the postures among this list: 'sitting', 'standing', 'lying down', 'reclining', 'upright'
                            - Only use the activities among this list: 'video gaming',  'walking', 'stair climbing',  'getting ready',  'driving',  'bicycling',  'vigorous bicycling',  'aerobics',  'cleaning',  'cooking',  'laundry',  'playing with pet',  'listening to music',  'watching movies/TV',  'studying',  'reading',  'riding in car',  'riding train',  'riding bus',  'playing musical instruments',  'attending meeting',  'computer using', 'phone using',  'running',  'getting dressed',  'grooming',  'using bathroom',  'eating',  'talking','strength training',  'washing dishes',  'carrying groceries',  'putting away groceries',  'shopping',  'making bed',  'packing/unpacking',  'sleeping',  'playing sports'
                            """

one_hours = ['8am-9am', '9am-10am', '10am-11am', '11am-12pm', '12pm-1pm', '1pm-2pm', '2pm-3pm', '3pm-4pm', '4pm-5pm', '5pm-6pm', '6pm-7pm', '7pm-8pm', '8pm-9pm', '9pm-10pm', '10pm-11pm']
def run_query_on_subject(subject_id, hours = one_hours):
    results = {}
    for hour in hours:
        initial_query = f"""On {participants[subject_id]}, can you tell me the list of postures and activities {subject_id} did from {hour}? 
        
        - Response with a list in the format of start time-end time: posture, activities. 
        - List activities for the entire one hour period. 
        - They can do multiple activities at once. Suggest all the activities they might be doing.
        - Make sure to include posture
        - Use dynamics time periods based on changes in activities and postures.
        - Check the start and end time of each activity and posture, and make sure they are correct TO THE MINUTE.
        
        Example:
        8:00-8:03: sitting, computer using
        8:03-8:08: standing, walking, phone using
        8:08-8:24: standing, cooking
        8:24-8:30: sitting, eating
        8:30-9:00: sitting, computer using, reading, phone using
        """
        sensemaker = sensemaking_process.SenseMaker(
                        initial_query,
                        presentation_format
                    )
        sensemaker.make_sense()
        results[hour] = {}

        answer = sensemaker.answer

        results[hour]["answer"] = sensemaker.answer
        results[hour]["action_plan"] = sensemaker.hypothesis
        results[hour]["memory"] = sensemaker.memory
        results[hour]["understanding"] = sensemaker.understanding
        results[hour]["information_requests"] = sensemaker.information_request
        results[hour]["function_calls"] = sensemaker.function_calls
        results[hour]["step_history"] = sensemaker.step_history
        # print(results[hour]["answer"])

    # create directory if not exists
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    # save results to pickle file
    with open(f'evaluation/GLOSS4HAR_results_{subject_id}_4h_nohelper.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

def run_individual_query(subject_id, memory = '', activities = "", activity='walking', time_period='9:05-9:10am'):
    query = get_triangulation_prompt_from_list_of_activities_no_time(
        subject_id=subject_id,
        # activity=activity,
        time_period=time_period,
        participants=participants,
        activities=activities
    )
    sensemaker = sensemaking_process.SenseMaker(
                    query, presentation_format
                )
    if memory != '' and not ABLATION_MEMORY:
        sensemaker.memory = "This is the previous annotations (have been correct). Do not include these in final answer:\n" + memory
    sensemaker.make_sense()

    return sensemaker

def run_individual_query_for_uEMA(subject_id, memory = '', activities = "", activity='walking', time_period='9:05-9:10am'):
    query = get_triangulation_prompt(
        subject_id=subject_id,
        # activity=activity,
        time_period=time_period,
        participants=participants,
    )
    sensemaker = sensemaking_process.SenseMaker(
                    query, presentation_format
                )
    if memory != '' and not ABLATION_MEMORY:
        sensemaker.memory = "This is the previous annotations (have been correct). Do not include these in final answer:\n" + memory
    sensemaker.make_sense()

    return sensemaker

def run_individual_query_for_correction(subject_id, memory = '', activities = "", activity='walking', time_period='9:05-9:10am'):
    query = get_correction_prompt(
        subject_id=subject_id,
        activity=activity,
        time_period=time_period,
        participants=participants,
    )
    sensemaker = sensemaking_process.SenseMaker(
                    query, presentation_format
                )
    if memory != '' and not ABLATION_MEMORY:
        sensemaker.memory = "This is the previous annotations (have been correct). Do not include these in final answer:\n" + memory
    sensemaker.make_sense()

    return sensemaker

def run_query_on_annotations(csv_file):
    df = pd.read_csv(csv_file)
    cleaned_csv_file = csv_file.replace('.csv', '_results_oss.csv')

    if os.path.exists(cleaned_csv_file):
        cleaned_df = pd.read_csv(cleaned_csv_file)
        for col in ('result', 'thinking'):
            if col in cleaned_df.columns:
                df[col] = cleaned_df[col]
    else:
        df['result'] = None
        df['thinking'] = None
        df.to_csv(cleaned_csv_file, index=False)

    memory_past_annotations = ''

    for index, row in df.iterrows():
        if pd.notna(df.at[index, 'result']):
            memory_past_annotations += f"{df.at[index, 'result']}\n"
            continue

        subject = row['subject']
        date = row['date']
        labels = row['labels']
        uncertainty_start = row['uncertainty_start']
        uncertainty_end = row['uncertainty_end']

        result = run_individual_query_for_correction(
            subject,
            memory=memory_past_annotations,
            activity=labels,
            time_period=f"{uncertainty_start}-{uncertainty_end}"
        )

        try:
            df.at[index, 'result'] = str(result.answer)
            df.at[index, 'thinking'] = str(result.understanding)
            memory_past_annotations += f"{df.at[index, 'result']}\n"
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            df.at[index, 'result'] = 'Error processing row'
            df.at[index, 'thinking'] = ''

        df.to_csv(cleaned_csv_file, index=False)

    return df

def run_query_on_task_uEMA(csv_file):
    # parse the csv link to get the subject_id: /mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3/pilot2_cleaned.csv
    subject = csv_file.split('/')[-1].split('_')[0]
    # create the name of the result file
    cleaned_csv_file = csv_file.replace('.csv', '_results_oss.csv')
    # if no file exists, create one 
    if not os.path.exists(cleaned_csv_file):
        cleaned_df = pd.DataFrame()
        # create the columns subject,date,uncertainty_start,uncertainty_end,labels,result,action_plan,memory,understanding,information_requests,function_calls
        cleaned_df['subject'] = ''
        cleaned_df['date'] = ''
        cleaned_df['uncertainty_start'] = ''
        cleaned_df['uncertainty_end'] = ''
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

    # get the subject, date, labels, uncertainty_start and uncertainty_end
    for hours in one_hours:
        time_period = hours

        # check if time period already exists in cleaned_df (uncertainty_start-uncertainty_end)
        if ((cleaned_df['subject'] == subject) & (cleaned_df['uncertainty_start'] == time_period.split('-')[0]) & (cleaned_df['uncertainty_end'] == time_period.split('-')[1])).any():
            continue

        # create the query
        result = run_individual_query_for_uEMA(
            subject,
            memory=memory_past_annotations,
            time_period=time_period
        )
        try:
            new_row = {
                'subject': subject,
                'date': participants[subject],
                'uncertainty_start': time_period.split('-')[0],
                'uncertainty_end': time_period.split('-')[1],
                'labels': '',  # no labels for this task
                'result': str(result.answer),
                'action_plan': str(result.hypothesis),
                'memory': str(result.memory),
                'understanding': str(result.understanding),
                'information_requests': str(result.information_request),
                'function_calls': str(result.function_calls)
            }
            cleaned_df = pd.concat([cleaned_df, pd.DataFrame([new_row])], ignore_index=True)

            # update the memory with the result
            memory_past_annotations += f"{new_row['result']}\n"
        except Exception as e:
            print(f"Error processing time period {time_period}: {e}")
        cleaned_df.to_csv(cleaned_csv_file, index=False)
    return cleaned_df


def run_query_on_task_lst(csv_file):
    # parse the csv link to get the subject_id: /mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3/pilot2_cleaned.csv
    subject = csv_file.split('/')[-1].split('_')[0]
    # create the name of the result file
    cleaned_csv_file = csv_file.replace('.csv', '_results_oss.csv')

    # read the csv file content into a string
    with open(csv_file, 'r') as f:
        activities = f.read()

    # if no file exists, create one 
    if not os.path.exists(cleaned_csv_file):
        cleaned_df = pd.DataFrame()
        # create the columns subject,date,uncertainty_start,uncertainty_end,labels,result,action_plan,memory,understanding,information_requests,function_calls
        cleaned_df['subject'] = ''
        cleaned_df['date'] = ''
        cleaned_df['uncertainty_start'] = ''
        cleaned_df['uncertainty_end'] = ''
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

    # get the subject, date, labels, uncertainty_start and uncertainty_end
    for hours in one_hours:
        time_period = hours

        # check if time period already exists in cleaned_df (uncertainty_start-uncertainty_end)
        if ((cleaned_df['subject'] == subject) & (cleaned_df['uncertainty_start'] == time_period.split('-')[0]) & (cleaned_df['uncertainty_end'] == time_period.split('-')[1])).any():
            continue

        try:
            # create the query
            result = run_individual_query(
                subject,
                memory=memory_past_annotations,
                time_period=time_period,
                activities=activities
            )   
            new_row = {
                'subject': subject,
                'date': participants[subject],
                'uncertainty_start': time_period.split('-')[0],
                'uncertainty_end': time_period.split('-')[1],
                'labels': '',  # no labels for this task
                'result': str(result.answer),
                'action_plan': str(result.hypothesis),
                'memory': str(result.memory),
                'understanding': str(result.understanding),
                'information_requests': str(result.information_request),
                'function_calls': str(result.function_calls)
            }
            cleaned_df = pd.concat([cleaned_df, pd.DataFrame([new_row])], ignore_index=True)

            # update the memory with the result
            memory_past_annotations += f"{new_row['result']}\n"
        except Exception as e:
            print(f"Error processing time period {time_period}: {e}")
        cleaned_df.to_csv(cleaned_csv_file, index=False)
    return cleaned_df


if __name__ == "__main__":

    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot2_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot5_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot6_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot7_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot8_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot9_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot10_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot11_cleaned.csv")

    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot2_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot5_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot6_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot7_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot8_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot9_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot10_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_presentation/pilot11_cleaned.csv")

    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot2_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot5_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot6_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot7_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot8_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot9_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot10_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_presentation/pilot11_cleaned.csv")

    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot2_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot5_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot6_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot7_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot8_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot9_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot10_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_presentation/pilot11_cleaned.csv")

    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot2_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot5_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot6_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot7_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot8_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot9_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot10_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_presentation/pilot11_cleaned.csv")

    # ABLATION_PRESENTATION_AGENT = False
    # ABLATION_MEMORY = True

    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot2_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot5_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot6_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot7_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot8_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot9_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot10_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task3_no_memory/pilot11_cleaned.csv")

    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot2_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot5_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot6_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot7_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot8_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot9_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot10_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task4_no_memory/pilot11_cleaned.csv")

    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot2_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot5_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot6_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot7_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot8_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot9_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot10_cleaned.csv")
    # run_query_on_task_lst("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task6_no_memory/pilot11_cleaned.csv")

    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot2_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot5_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot6_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot7_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot8_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot9_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot10_cleaned.csv")
    # run_query_on_task_uEMA("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2_no_memory/pilot11_cleaned.csv")

    run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot2_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot5_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot6_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot7_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot8_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot9_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot10_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean_no_memory/pilot11_cleaned.csv")

