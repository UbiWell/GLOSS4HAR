import os
import sys

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.data_driver import run_function_from_dict, json_to_dict, extract_data_multiple_type
from gpt_utils import generate_function_calling_prompt
import json

import data_streams.android_location as android_location
import data_streams.android_phone_usage as app_usage_data
import data_streams.pixel_ambient_noises as ambient_noise_data
import data_streams.pixel_uEMA as uEMA
import data_streams.pixel_steps_data as phone_steps_data
import data_streams.pixel_hr as heart_rate_data
import data_streams.pixel_skin_temp as skin_temp_data
import data_streams.pixel_wear_detection as wear_detection_data
import data_streams.pixel_wrist_auc as wrist_auc_data

from agents.constants import databases
from data_streams.generic_coding_functions import GenericCodingFunctions
from agents.constants import USE_AZURE, USE_GPT5, USE_GPT_OSS, USE_UEMA
from agents.gpt_oss_langchain import GPTOSSChatModel
GPTOSSChatModel.model_rebuild()

if not USE_AZURE and not USE_GPT5 and not USE_GPT_OSS:
    llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                    model_name="gpt-4o",
                    temperature= 0)
elif USE_GPT_OSS:
    llmchat = GPTOSSChatModel()
elif USE_GPT5:
    llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                    model_name="gpt-5")


class GenericDatabaseManager:
    def __init__(self):
        self.database_chain = self.database_agent()
        self.function_call_history = []
        self.coding_function = None

    def invoke(self, input_params):

        req_databases = input_params["databases"]
        calling_functions = {}
        for database in req_databases:
            if " database" not in database:
                database = database + " database"
        print(f"Databases requested: {req_databases}")
        for database in req_databases:
            if database == "location database": # or database == "location":
                calling_functions.update(android_location.functions)
            elif database == "uEMA database" and USE_UEMA:
                calling_functions.update(uEMA.functions)
            elif database == "heart rate database": # or database == "heart rate":
                calling_functions.update(heart_rate_data.functions)
            elif database == "phone usage database":# or database == "phone usage":
                calling_functions.update(app_usage_data.functions)
            elif database == "ambient noise database":# or database == "ambient noise":
                calling_functions.update(ambient_noise_data.functions)
            elif database == "step count database": # or database == "step count":
                calling_functions.update(phone_steps_data.functions)
            elif database == "skin temperature database":# or database == "skin temperature":
                calling_functions.update(skin_temp_data.functions)
            elif database == "watch wear database": # or database == "watch wear":
                calling_functions.update(wear_detection_data.functions)
            else:
                print(f"Database not recognized: {database}. Please check the database name.")
                return {"NOT POSSIBLE": "not possibe to answer with gven databases code"}

        functions = {}

        coding_functions_obj = GenericCodingFunctions(calling_functions, req_databases)
        cfs = coding_functions_obj.coding_functions
        functions.update(cfs)
        self.coding_function = coding_functions_obj.get_results_through_data_computation
        functions.update(calling_functions)

        input_instructions = """ You are manager for following databse: \n"""
        for d in req_databases:
            input_instructions += d + ": " + databases[d]["info"] + "\n"
            if "additional_instructions" in databases[d]:
                input_instructions += "additional instructions: "
                input_instructions += databases[d]["additional_instructions"]

        input_instructions += "The users will request you for data. You have following functions that you can call to fulfil the request of the user: \n"

        output_instructions = '''
                Your task is to return the dict of function calls with input params needed to answer the question asked by the user. 
                Do not call the functions with exact same parameters if they have been already called in the previous function calls

             Output Format: {function ID1: {"name": function name, "params" : {param1: value1, param2: value2}},
                            function ID2: {"name": function name, "params" :{param1: value1, param2: value2}}}
                            
            Example: {"CODING1": {"name": "get_results_through_data_computation", "params": {"user_query": "how many steps user pilot2 did?", "start_time": "2024-07-09 00:00:00", "end_time": "2024-07-09 23:59:59"}}}


                            Instructions:
                            1) Just return the dict {} of function calls with input params in the required order. do not return anything else in the output including ```json or ```python.
                            2) return {"NOT POSSIBLE" : reason} if the question cannot be answered with the existing functions or you have already called all the functions that could have answered the query.
                            3) Do not write any additional code, just the list of function calls with input params
                            4) The dates in params should be in "%Y-%m-%d %H:%M:%S" format
                            5) Only add function calls when you know exact values of the parameters for the function call. do not add placeholders as parameters.  
                            7) Do not assume that you do not have data in the databases for requested days unless you have already tried fetching the data. It has nothing to do with your training data.
                            

                '''

        prompt = generate_function_calling_prompt(input_instructions, cfs, output_instructions)
        # print("INPUT PARAMS: ", input_params    )
        # create another version of input param without the databases field
        input_params_no_db = {k: v for k, v in input_params.items() if k != "databases"}
        info_chain = self.database_chain.invoke(
            {'user_query': input_params_no_db, 'function_call_history': self.function_call_history,
             'prompt': prompt})

        return info_chain

    def extract_data_step(self, chain_output):
        calls = json.loads(chain_output.content)
        if ("NOT POSSIBLE" in calls or "TERMINATE" in calls):
            return calls
        return extract_data_multiple_type(chain_output, self.coding_function)

    def database_agent(self):
        prompt = """{prompt}"""

        template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("user", "User Query: {user_query}\n\n Previous Function Calls: {function_call_history}")
            ]
        )

        chain = (template | llmchat | self.extract_data_step)
        return chain


if __name__ == "__main__":
    question = (
        "List periods of time user_id 'pilot2' took steps on 2025-02-18.")

    response = GenericDatabaseManager().invoke({'user_query': question, 'databases': ["step count database"]})

    print(response)
