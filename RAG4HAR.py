import random
import sys
import os
import pickle
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))

import sensemaking_process
from data_processing.data_processing_utils import fetch_first_and_last_document, fetch_documents_between_timestamps
from data_streams.constants import IOS_LOCATION, GARMIN_HR, GARMIN_STRESS
from datetime import datetime, timedelta
import pandas as pd
from agents.gpt_utils import invoke_with_retry
from agents.constants import databases

from data_streams.android_location import *
from data_streams.android_phone_usage import *
from data_streams.pixel_ambient_noises import *
from data_streams.pixel_hr import *
from data_streams.pixel_skin_temp import *
from data_streams.pixel_steps_data import *
from data_streams.pixel_uEMA import *
from data_streams.pixel_wear_detection import *

from typing import Optional
import requests

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

class Config:
    """Configuration class for the Flask LLM Chat application"""
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # LLM API configuration
    LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:11434/api/generate')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-oss:20b')
    
    # LLM API parameters
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '8000'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0'))
    
    # System message for the LLM
    SYSTEM_MESSAGE = os.getenv('SYSTEM_MESSAGE', "You are a helpful assistant.")
    
    # Timeout settings
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '240'))
    STATUS_CHECK_TIMEOUT = int(os.getenv('STATUS_CHECK_TIMEOUT', '10'))
    
    @classmethod
    def get_llm_headers(cls) -> dict:
        """Get headers for LLM API requests"""
        headers = {
            'Content-Type': 'application/json'
        }
            
        return headers
    
    @classmethod
    def validate_config(cls) -> list:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        if cls.LLM_API_URL == 'http://localhost:8000/v1/chat/completions':
            warnings.append("Using default localhost LLM API URL. Make sure your LLM server is running.")
        
        if cls.SECRET_KEY == 'your-secret-key-change-this-in-production':
            warnings.append("Using default secret key. Change SECRET_KEY in production.")
        
        return warnings 

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

def run_query_on_gpt_oss(query: str, model: Optional[str] = None) -> dict:
    """
    Run the query through the LLM API and return the response.
    """
    config = Config()

    llm_request_data = {
        'model': model or config.LLM_MODEL,
        'prompt': query,
        'stream': False
    }
    headers = config.get_llm_headers()

    # Make request to the remote LLM API
    response = requests.post(
        config.LLM_API_URL,
        json=llm_request_data,
        headers=headers,
        timeout=config.REQUEST_TIMEOUT
    )

    if response.status_code != 200:
        raise Exception(f"LLM API request failed with status code {response.status_code}: {response.text}")

    resp = response.json()
    
    return resp["response"], resp["thinking"]

def run_query(query):
    """
    Run the query through GPT 4
    """
    model = lcai.ChatOpenAI(model="gpt-4o", temperature=0)

    resp = model.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ])

    return resp.content

def run_individual_query(subject_id, activity, time_period, start_time, stop_time, model="gpt4"):
    """
    Run the individual query for the subject_id with the given activity and time_period.
    """
    # create the vanilla query
    initial_query = create_vanilla_query(subject_id, activity, time_period, start_time, stop_time)

    if model == "gpt4":
        # run the query through GPT 4
        response = run_query(initial_query)
        return response
    elif model == "gpt-oss":
        # run the query through GPT OSS
        response, thinking = run_query_on_gpt_oss(initial_query, model="gpt-oss:20b")
        print(f"Thinking: {thinking}")
        print(f"Response: {response}")
        return response, thinking

def run_query_on_annotations(csv_file):
    df = pd.read_csv(csv_file)
    
    # loop through each row in the dataframe
    for index, row in df.iterrows():
        # get the subject, date, labels, uncertainty_start and uncertainty_end
        subject = row['subject']
        date = row['date']
        labels = row['labels']
        uncertainty_start = row['uncertainty_start']
        uncertainty_end = row['uncertainty_end']

        response, thinking = run_individual_query(
            subject,
            labels,
            f"{uncertainty_start}-{uncertainty_end}",
            uncertainty_start,
            uncertainty_end,
            model="gpt-oss"
        )

        df.at[index, 'result'] = response
        df.at[index, 'thinking'] = thinking

        # save the dataframe to a new csv file
        cleaned_csv_file = csv_file.replace('.csv', '_results_vanilla_oss.csv')
        df.to_csv(cleaned_csv_file, index=False)
    return df

if __name__ == "__main__":
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot2_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot8_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot5_cleaned.csv")
    # run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot6_cleaned.csv")
    run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot7_cleaned.csv")
    run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot9_cleaned.csv")
    run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot10_cleaned.csv")
    run_query_on_annotations("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/pilot11_cleaned.csv")
