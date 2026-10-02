from agents.constants import databases
import random
import os
import langchain_openai as lcai
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from data_processing.data_processing_utils import fetch_first_and_last_document
from data_streams.constants import IOS_LOCATION
from datetime import datetime, timedelta
import pandas as pd

uids = ["test007", "test008", "test009", "test011", "u008",	"u009",	"u010", "u011", "u012", "u013", "u014"]
days = [1, 2, 7]
datasets_numbers = [1, 2, 3]

number_of_queries = 10
from agents.constants import USE_AZURE, USE_GPT5
from agents.gpt_oss_langchain import GPTOSSChatModel
GPTOSSChatModel.model_rebuild()

if not USE_AZURE and not USE_GPT5 and not USE_GPT_OSS:
    llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                    model_name="gpt-4o",
                    temperature= 0)
elif USE_GPT_OSS:
    llmchat = lcai.GPTOSSChatModel()
elif USE_GPT5:
    llmchat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                    model_name="gpt-5")
else:
    llmchat = lcai.AzureChatOpenAI(
                    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
                    azure_deployment="PROPILOT",
                    openai_api_version="2024-05-01-preview",
                    model_name="gpt-4o",
                    temperature=0)


class Output(BaseModel):
    objective_query: str = Field(description="objective query")


def generate_query_prompt(uid, first_day, last_day, dataset_number):
    if first_day == last_day:
        first_day = datetime.strptime(first_day, "%Y-%m-%d %H:%M:%S").date()
        first_day = first_day.strftime("%Y-%m-%d")
        task = f""" Task: You have to generate a Objective Query for user_id as {uid}  on {first_day} using these {dataset_number} databases.
                """
    else:
        task = f""" Task: You have to generate a Objective Query for user_id as {uid} between {first_day} and {last_day} using these {dataset_number} databases."""

    prompt = '''
    Objective queries are characterized by clearly defined responses. For example, the question,
    "Between 2024-05-29 00:00:00 and 2024-06-06 00:00:00 did test004 exceed 5,000 steps?" constitutes a specific, tractable query.
    '''

    prompt += task

    for i in range(0, dataset_number):
        random_key = random.choice(list(databases.keys()))
        random_value = databases[random_key]
        prompt += f'''
        Dataset {i + 1}:
        {random_key}
        {random_value}
        '''

    format_instructions = """
    Format
    Instructions:
    - you query should involve all the databases mentioned above.
    - The objective query should be in the form of a question.
    -  Just return the objective query in form of json.
    {"objective_query": "your query"}
    - do not include  any other information in the response.
    - do not include ```json or ```python in the response.
    """
    prompt += format_instructions
    if first_day == last_day:
        example = """
        Example of objective query
        for user_id as test004 on 2024-05-29 using these 2 databases.The query should include all the below datasets.
        Dataset1: wifi database
        {
        'info': 'Contains data whether phone is connected to wifi or not. It also contains the wifi name to which phone is connected.',
        'device': 'Phone'
        }
    
    
        Dataset2: garmin stress database
        {'info': 'Contains stress predictions (0 to 1) from ibi data recorded from the Garmin smartwatch.',
         'device': 'Garmin Smartwatch'}
        
        {objective_query: On 2024-05-29 for test004 what was the wifi name when person had high stress prediction( > 0.8)?}"
        """
    else:
        example = """
                Example of objective query
                for user_id as test004 on 2024-05-29 00:00:00 and 2024-06-03 00:00:00 using these 2 databases.The query should involve accessing/triangulating data from all the below datasets.
                Dataset1: wifi database
                {
                'info': 'Contains data whether phone is connected to wifi or not. It also contains the wifi name to which phone is connected.',
                'device': 'Phone'
                }


                Dataset2: garmin stress database
                {'info': 'Contains stress predictions (0 to 1) from ibi data recorded from the Garmin smartwatch.',
                 'device': 'Garmin Smartwatch'}

                {objective_query: Betwen 2024-05-29 00:00:00 and 2024-06-03 for test004 what was the wifi name when person had high stress prediction( > 0.8)?}"
                {objective_query: Between 29th March and 1st April  what was the wifi name when test004 had high stress prediction( > 0.8)?}"
                """

    prompt += example

    return prompt


class QueryGeneratorAgent:

    def __init__(self):
        self.query_chain = self.query_agent()
        self.function_call_history = []


    def invoke(self, input_params):
        info_chain = self.query_chain.invoke({'prompt': input_params['prompt']})
        return info_chain


    def query_agent(self):
        parser = JsonOutputParser(pydantic_object=Output)

        prompt = """
        {prompt}
        """

        template = PromptTemplate(
            template=prompt,
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = template | llmchat | parser
        return chain


queries = []

def main():
    for i in range(0, number_of_queries):
        u = random.choice(uids)
        day = random.choice(days)
        dataset_number = random.choice(datasets_numbers)

        first, last = fetch_first_and_last_document(u, IOS_LOCATION)

        first_date = datetime.fromtimestamp(first['timestamp']).date()
        last_date = datetime.fromtimestamp(last['timestamp']).date()


        first_date_ = first_date + timedelta(days=day - 1)

        # Calculate the number of days between the two dates
        delta_days = (last_date - first_date_).days

        if delta_days < 0:
            continue

        # Choose a random day within the range
        chosen_last_date = first_date_ + timedelta(days=random.randint(0, delta_days))
        chosen_first_date = chosen_last_date - timedelta(days=day - 1)

        chosen_last_date = chosen_last_date.strftime("%Y-%m-%d %H:%M:%S")
        chosen_first_date = chosen_first_date.strftime("%Y-%m-%d %H:%M:%S")

        prompt = generate_query_prompt(u, chosen_first_date, chosen_last_date, dataset_number)
        response = QueryGeneratorAgent().invoke({'prompt': prompt})
        q = {"user_id": u, "first_date": chosen_first_date, "last_date": chosen_last_date, "query": response['objective_query'], "dataset_number": dataset_number}
        queries.append(q)

    df = pd.DataFrame(queries)

    csv_file_path = 'eval_queries.csv'
    df.to_csv(csv_file_path, index=False)

def get_dates():
    for u in uids:
        first, last = fetch_first_and_last_document(u, IOS_LOCATION)

        # print(first)
        # print(last)

        first_date = datetime.fromtimestamp(first['timestamp']).date()
        last_date = datetime.fromtimestamp(last['timestamp']).date()
        print(u, first_date, last_date)



get_dates()
