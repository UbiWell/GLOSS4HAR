import os
import sys

import json
from datetime import datetime
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data_streams.android_location as android_location
import data_streams.android_phone_usage as android_phone_usage
import data_streams.pixel_ambient_noises as pixel_ambient_noises
import data_streams.pixel_hr as pixel_hr
import data_streams.pixel_skin_temp as pixel_skin_temp
import data_streams.pixel_steps_data as pixel_steps_data
import data_streams.pixel_uEMA as pixel_uEMA
import data_streams.pixel_wear_detection as pixel_wear_detection
import data_streams.pixel_wrist_auc as pixel_wrist_auc

stream_modules = [android_location, android_phone_usage, pixel_ambient_noises, pixel_hr, pixel_skin_temp,
                  pixel_steps_data, pixel_uEMA, pixel_wear_detection, pixel_wrist_auc]

all_functions = {}
for module in stream_modules:
    all_functions.update(module.functions)


def run_function_from_dict(function_name, params):
    try:
        for module in stream_modules:
            if hasattr(module, function_name):
                return getattr(module, function_name)(**params)
        print(f"Function {function_name} not found in any data stream module")
    except Exception as e:
        print(f"An error occurred: {e}")


def json_to_dict(json_string):
    # Convert JSON string to dictionary
    data_dict = json.loads(json_string)
    return data_dict


def get_function_description(functions, function_name):
    for key, function in functions.items():
        if function['name'] == function_name:
            return function['description']

    return "Function not found."

def extract_data_multiple_type(chain_output, coding_function = None):
    final_results = []
    chain_output = chain_output.content
    dict_output = json_to_dict(chain_output)
    for d in dict_output:
        func_name = dict_output[d]['name']
        function_id = d
        if "CODING" in function_id:
            if coding_function:
                params = dict_output[d]['params']
                if "start_time" in params and "end_time" in params:
                    # add the start and stop time to the user query
                    query = params['user_query'] + f' with start time {params["start_time"]} and end time {params["end_time"]}'
                    params['user_query'] = query
                query = params['user_query']
                final_results.append({"func": dict_output[d], "result": coding_function(query), "func_id": d})
                continue

        res = run_function_from_dict(func_name, dict_output[d]['params'])
        final_results.append({"func": dict_output[d], "result": res, "func_id": d})
    return final_results
