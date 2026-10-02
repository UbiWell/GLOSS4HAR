import os
import json

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from agents.constants import databases
from agents.data_driver import extract_data, extract_data_multiple_type, get_function_description
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents.gpt_utils import generate_function_calling_prompt
from agents.constants import USE_AZURE, USE_GPT5, USE_GPT_OSS
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

import data_streams.activity_data as activity_data
import data_streams.location_data as location_data
import data_streams.phone_steps_data as phone_steps_data
import data_streams.garmin_hr_data as heart_rate_data
import data_streams.lock_unlock_data as lock_unlock_data
import data_streams.garmin_steps_data as garmin_steps_data
import data_streams.wifi_data as wifi_data
import data_streams.app_usage_data as app_usage_data
import data_streams.battery_data as battery_data
import data_streams.call_log as call_log
import models.stress_prediction_model as stress

all_functions = {**activity_data.functions, **location_data.functions, **phone_steps_data.functions,
                 **heart_rate_data.functions, **lock_unlock_data.functions, **garmin_steps_data.functions,
                 **wifi_data.functions, **app_usage_data.functions, **battery_data.functions, **call_log.functions,
                 **stress.functions}

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def generate_prompt():
    prompt = "You have data from following databases: \n"

    databases_section = "\n".join(
        f"    {idx + 1}. {db_name}  \n"
        f"       - Info: {details['info']}  \n"
        f"       - Device: {details['device']}  \n"
        for idx, (db_name, details) in enumerate(databases.items())
    )
    prompt += databases_section

    prompt += "You can call following functions to fetch data from databases: \n"

    input_instructions = """
            The users will request you for data and analysis.

             You have following functions that you can call to fulfil the request of the user.

             Only call functions if request is can be answered using data in the databases otherwise return
             {"NOT POSSIBLE" : reason}
             
             Use Chat History to understand what data you already have and what additional data you need to fetch.
             
             if the all data needed to answer the question is available then return answer to user query in natural language
             {"TERMINATE" : answer to the query}
             
             """

    output_instructions = \
        """
            Your task is to return the dict of function calls with input params needed to answer the question asked by the user. 
            Do not call the functions with exact same parameters if they have been already called in the previous function calls.


            Output Format: 
                       {
                          Function ID: {
                            "name": "function name",
                            "params": {{
                              "param1": "value1",
                              "param2": "value2"
                            }
                          },
                          Function ID: {
                            "name": "function name",
                            "params": {
                              "param1": "value1",
                              "param2": "value2"
                            }
                          }
                        }
                        
                Take Function ID, function name of relevant functions
                
                Instructions:
                1) Follow the output format mention above of function calls with input params in the required order. do not return anything else in the output including ```json or ```python.
                2) Be optimistic that one of your functions can answer these question.
                2) return {"NOT POSSIBLE" : reason} if the question cannot be answered with the existing functions (or database) or you have already called all the functions that could have answered the query.
                4) Do not write any additional code, just the list of function calls with input params
                5) The dates in params should be in "%Y-%m-%d %H:%M:%S" format
                7) Do not call the functions with exact same parameters if they have been already called in the previous function calls.
                8) if the all data needed to answer the question is available then return
                    {"TERMINATE" : answer to the query}
                """

    prompt += generate_function_calling_prompt(input_instructions, all_functions, output_instructions)

    return prompt


def generate_answering_prompt(results, query):
    prompt = f"""
    User asked following questions:
    {query}
    
    You called following functions to fetch data:
    """

    for res in results:
        f_name = res['func']['name']
        params = res['func']['params']
        desc = get_function_description(all_functions, f_name)
        func_results = res['result']

        prompt += f"Function Name: {f_name}\n"
        prompt += f"Parameters: {params}\n"
        prompt += f"Description: {desc}\n"
        prompt += f"Results: {func_results}\n\n"

    prompt += """
    Your task is to generate the answer to the user query based on the data fetched from the functions called.
    Present the results in natural language.
    """

    return prompt


class AllInOneAgent:
    def __init__(self):
        self.llm_chain = self.all_in_one_agent()
        self.answering_chain = self.answering_agent()

    def extract_data_step(self, chain_output):
        calls = json.loads(chain_output.content)
        if ("NOT POSSIBLE" in calls or "TERMINATE" in calls):
            return calls
        return extract_data_multiple_type(chain_output)

    def all_in_one_agent(self):
        prompt = """{prompt}"""
        template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("user", "User Query: {user_query}\n"),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )
        chain = template | llmchat | self.extract_data_step

        return chain

    def answering_agent(self):
        prompt = """{prompt}"""

        template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt)
            ]
        )
        chain = template | llmchat | StrOutputParser()
        return chain

    def invoke(self, input_params):
        prompt = generate_prompt()
        info_chain = self.llm_chain.invoke({'user_query': input_params['user_query'], 'prompt': prompt, 'chat_history': input_params['chat_history']})
        return info_chain

    def invoke_answering(self, input_params):
        answering_prompt = generate_answering_prompt(input_params['results'], input_params['user_query'])
        info_chain = self.answering_chain.invoke({'prompt': answering_prompt})
        return info_chain

    def run_agent(self, user_query):
        run_flag = True
        chat_history = []
        while (run_flag):
            results = self.invoke({'user_query': user_query, 'chat_history': chat_history})
            if ("NOT POSSIBLE" in results):
                return "the question cannot be answered with the existing data."
            if ("TERMINATE" in results):
                return results["TERMINATE"]
            else:
                for r in results:
                    chat_history.extend([HumanMessage(content=str(r['func'])), str(r["result"])])
            # answer = self.invoke_answering({'results': results, 'user_query': user_query})
            # return answer
        # answer = self.invoke_answering({'results': results, 'user_query': user_query})


if __name__ == "__main__":
    agent = AllInOneAgent()
    user_query = "counts of different kinds of activities when  07/20/2024?"
    answer = agent.run_agent(user_query)
    print(answer)
