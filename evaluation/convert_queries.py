from agents.constants import databases
import random
import sys
import os
import pickle
import sensemaking_process
import langchain_openai as lcai
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

from data_processing.data_processing_utils import fetch_first_and_last_document, fetch_documents_between_timestamps
from data_streams.constants import IOS_LOCATION, GARMIN_HR, GARMIN_STRESS
from datetime import datetime, timedelta
import pandas as pd
from agents.gpt_utils import invoke_with_retry

uids = ["test007", "test008", "test009", "test011", "test006", "u008", "u009", "u010", "u011", "u012", "u013", "u014"]


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
    updated_query: str = Field(description="updated query")


def generate_query_prompt(query, user, date):
    prompt = f''' Your job is to replace start date and user_id in given query
    query: {query}
    uid: {user}
    start date: {date}
    
    You have to keep the date format as used in original query
    You have to keep same different between start and end date as in original query
    
    just return the query as {{updated_query: updated query}}
    
    Examples:
    query: How many unique places did test007 travel to in the afternoon (12 PM - 6 PM) on 2024-07-08?
    uid: test007
    start date: 2024-05-29 00:00:00
    {{"updated_query": "How many unique places did test007 travel to in the afternoon (12 PM - 6 PM) on 2024-05-29?"}}
    
    query: On Jan 8 2025, 2025, which Wi-Fi network was test007 connected to when their heart rate went over 120 bpm? could it have been during an exciting or intense moment?
    uid: u012
    start date: 2024-11-29 00:00:00
    {{"updated_query": "On Nov 29 2024, which Wi-Fi network was u012 connected to when their heart rate went over 120 bpm? could it have been during an exciting or intense moment?"}}
    
    query: How many total incoming and outgoing calls did test007 make from 2024-07-01 to 2024-07-08?
    uid: u011
    start date: 2024-03-16 00:00:00
    {{"updated_query": "How many total incoming and outgoing calls did u011 make from 2024-03-16 to 2024-03-23?"}}
    
    '''
    return prompt


class QueryGeneratorAgent:

    def __init__(self):
        self.query_chain = self.query_agent()

    def invoke(self, input_params):
        prompt = generate_query_prompt(input_params['query'], input_params['user'], input_params['date'])
        info_chain = self.query_chain.invoke({'prompt': prompt})
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



def get_dates():
    for u in uids:
        first, last = fetch_first_and_last_document(u, IOS_LOCATION)

        # print(first)
        # print(last)

        first_date = datetime.fromtimestamp(first['timestamp']).date()
        last_date = datetime.fromtimestamp(last['timestamp']).date()
        print(u, first_date, last_date)


# get_dates()



def get_data_report(uid):
    first, last = fetch_first_and_last_document(uid, IOS_LOCATION)
    first_date = datetime.fromtimestamp(first['timestamp'])
    last_date = datetime.fromtimestamp(last['timestamp'])

    compliance = {}

    start_date = first_date
    while start_date < last_date:
        end_date = start_date + timedelta(days=1)
        gps_records = fetch_documents_between_timestamps(uid, start_date.timestamp(), end_date.timestamp(), IOS_LOCATION)

        gps_count = 0
        previous_time = 0
        gps_minutes = 0
        for gps in gps_records:
            if gps['timestamp'] - previous_time < 15 * 60:
                gps_minutes += float(gps['timestamp'] - previous_time) / 60
                gps_count += 1
            previous_time = gps['timestamp']
        # gps_minutes = gps_count * 10

        empatica_count = len(fetch_documents_between_timestamps(uid, start_date.timestamp(), end_date.timestamp(), GARMIN_HR))
        stress_count = len(fetch_documents_between_timestamps(uid, start_date.timestamp(), end_date.timestamp(), GARMIN_STRESS))
        float(empatica_count) / (2 * 60), float(stress_count) / (6 * 60)
        date_string = start_date.strftime("%Y-%m-%d")
        compliance[date_string] = {'gps': float(gps_minutes) / 60, 'garmin_worn' : float(empatica_count) / (2 * 60), 'garmin_on' : float(stress_count) / (6 * 60)}
        start_date = end_date

    return compliance



def generate_compliance_reports():
    u = 0
    t = 0
    for uid in uids:
        compliance = get_data_report(uid)
        df = pd.DataFrame(compliance).T
        u += 1
        t += len(df)
    print("Total users: ", u)
    print("Total days: ", t)

    # df.to_csv(f"compliance_records/{uid}_compliance.csv", index=True)

generate_compliance_reports()
def find_relevant_date(row_count=2, pick_garmin_data=False, pick_app_date=False):
    all_compliance = []  # List to store each user's compliance DataFrame
    for uid in uids:
        # Read and process each user's compliance CSV
        df_compliance = pd.read_csv(f"compliance_records/{uid}_compliance.csv")
        df_compliance.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        df_compliance['uid'] = uid
        all_compliance.append(df_compliance)  # Append the processed DataFrame

    # Concatenate all user DataFrames into a single DataFrame
    stacked_compliance = pd.concat(all_compliance, ignore_index=True)

    if pick_garmin_data:
        # Pick the Garmin data
        stacked_compliance = stacked_compliance[stacked_compliance['garmin_worn'] > 10]
    if pick_app_date:
        # Pick the app data
        stacked_compliance = stacked_compliance[
            stacked_compliance['uid'].isin(['test007', 'test008', 'test009', 'test011', 'test006'])]

    sample_df = stacked_compliance.sample(n=row_count)
    print(set(stacked_compliance["uid"]))

    return list(zip(sample_df['uid'], sample_df['date']))


# print(find_relevant_date(row_count=2, pick_garmin_data=True, pick_app_date=False))




def convert_human_queries():
    queries = []
    uids = []
    start_dates = []
    labels = []
    df = pd.read_csv("human_queries_with_labels.csv")
    for i, row in df.iterrows():
        query = row["queries"]
        garmin = row["garmin"]
        label = row["query_type_combined"]
        app = row["app"]
        if garmin == "y" and app == 'y':
            dates = find_relevant_date(row_count=2, pick_garmin_data=True, pick_app_date=True)
        elif garmin == "y":
            dates = find_relevant_date(row_count=2, pick_garmin_data=True, pick_app_date=False)
        elif app == "y":
            dates = find_relevant_date(row_count=2, pick_garmin_data=False, pick_app_date=True)
        else:
            dates = find_relevant_date(row_count=2, pick_garmin_data=False, pick_app_date=False)

        for user, date in dates:
            response = invoke_with_retry(QueryGeneratorAgent(), 'invoke', {'user': user, 'date': date, 'query': query})
            queries.append(response['updated_query'].lower())
            uids.append(user)
            start_dates.append(date)
            labels.append(label)



    df_new = pd.DataFrame()
    df_new["updated_queries"] = queries
    df_new["uids"] = uids
    df_new["dates"] = start_dates
    df_new['labels'] = labels
    df_new.to_csv("updated_human_queries_with_labels_iter_1.csv", index=False)

# convert_human_queries()
def divide_queries_and_add_query_id():
    df = pd.read_csv("updated_human_queries_with_labels_iter_1.csv")
    obj_df = df[df['labels'].isin(['objective', 'mixed'])]
    subj_df = df[df['labels'].isin(['subjective', 'mixed'])]


    obj_df['query_id'] = ["obj_query_" + str(i) for i in range(1, len(obj_df) + 1)]
    subj_df['query_id'] = ["subj_query_" + str(i) for i in range(1, len(subj_df) + 1)]

    obj_df.to_csv("objective_human_queries_with_labels_iter_1.csv", index=False)
    subj_df.to_csv("subjective_human_queries_with_labels_iter_1.csv", index=False)

# divide_queries_and_add_query_id()

def get_stats():
    df = pd.read_csv("updated_human_queries_with_labels_iter_1.csv")
    df_old = pd.read_csv("human_queries_with_labels.csv")
    print(df_old['query_type_combined'].value_counts())


    print("after conversion")
    print(df['labels'].value_counts())


get_stats()
# def clear_dict(name):
#     with open(name, "wb") as file:
#         pickle.dump({}, file)
def run_objective_evaluations_with_sensemaking(start = 0, end = -1, name="results.pkl", rerun=False):
    df = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
    # df = df[:30]
    # df = df[df['labels'] == 'objective']
    queries = df['updated_queries']
    end = len(queries) if end == -1 else  end
    queries_to_run = queries[start:end]
    query_id_to_run = df['query_id'][start:end]

    if not os.path.exists(name):
        with open(name, "wb") as file:
            pickle.dump({}, file)
        print(f"{name} created as an empty dictionary.")

    with open(name, "rb") as file:
        results_dict = pickle.load(file)

    with open(os.devnull, 'w') as devnull:
        try:
            for query_id, query in zip(query_id_to_run, queries_to_run):
                print(f"Query: {query}")
                print(f"Running query: {query_id}")
                # if "stress" in query or "spending time among other people" in query:
                #     print("Skipping stress query")
                #     continue
                if results_dict.get(query_id) and 'answer' in results_dict[query_id]['run_1']:
                    print("Query already run")
                    if(rerun):
                        print("Rerunning query")
                        run_number = max([int(r.split("_")[1]) for r in results_dict[query_id]]) + 1
                        results_dict[query_id][f"run_{run_number}"] = {}
                    else:
                        continue
                else:
                    results_dict[query_id] = {}
                    results_dict[query_id]["run_1"] = {}

                sensemaker = sensemaking_process.SenseMaker(
                    query,
                    "answer clearly and concisely"
                )
                # sys.stdout = devnull
                sensemaker.make_sense()
                # sys.stdout = sys.__stdout__
                results_dict[query_id]['run_1']["answer"] = sensemaker.answer
                results_dict[query_id]['run_1']["action_plan"] = sensemaker.hypothesis
                results_dict[query_id]['run_1']["memory"] = sensemaker.memory
                results_dict[query_id]['run_1']["understanding"] = sensemaker.understanding
                results_dict[query_id]['run_1']["information_requests"] = sensemaker.information_request
                results_dict[query_id]['run_1']["function_calls"] = sensemaker.function_calls
                results_dict[query_id]['run_1']["step_history"] = sensemaker.step_history
                with open(name, "wb") as file:
                    pickle.dump(results_dict, file)
                print("Saving dictionary")


        except Exception as e:
            print("Error: ", e)
            with open(name, "wb") as file:
                pickle.dump(results_dict, file)
            print("Dictionary has been pickled and saved")


# run_objective_evaluations_with_sensemaking(start=0, end=-1, name=f"gloss_obj_iter_1_run_2.pkl")
# run_objective_evaluations_with_sensemaking(start=0, end=-1, name="gloss_obj_run_3.pkl")

# with open("results_obj_run1.pkl", "rb") as file:
#     results_dict = pickle.load(file)
#     print(results_dict)

def run_subjective_evaluations_with_sensemaking(start = 0, end = -1, name="results.pkl", rerun=False):
    df = pd.read_csv("subjective_human_queries_with_labels_iter_1.csv")
    # df = df[df['labels'] == 'objective']
    queries = df['updated_queries']
    end = len(queries) if end == -1 else end
    queries_to_run = queries[start:end]
    query_id_to_run = df['query_id'][start:end]

    if not os.path.exists(name):
        with open(name, "wb") as file:
            pickle.dump({}, file)
        print(f"{name} created as an empty dictionary.")

    with open(name, "rb") as file:
        results_dict = pickle.load(file)

    with open(os.devnull, 'w') as devnull:
        try:
            for query_id, query in zip(query_id_to_run, queries_to_run):
                print(f"Query: {query}")
                print(f"Running query: {query_id}")
                # if "stress"  in query:
                #     print("Skipping stress query")
                #     continue
                if results_dict.get(query_id):
                    print("Query already run")
                    if(rerun):
                        print("Rerunning query")
                        run_number = max([int(r.split("_")[1]) for r in results_dict[query_id]]) + 1
                        results_dict[query_id][f"run_{run_number}"] = {}
                    else:
                        continue
                else:
                    results_dict[query_id] = {}
                    results_dict[query_id]["run_1"] = {}

                sensemaker = sensemaking_process.SenseMaker(
                    query,
                    "Explain clearly and in details"
                )
                # sys.stdout = devnull
                sensemaker.make_sense()
                # sys.stdout = sys.__stdout__
                results_dict[query_id]['run_1']["answer"] = sensemaker.answer
                results_dict[query_id]['run_1']["action_plan"] = sensemaker.hypothesis
                results_dict[query_id]['run_1']["memory"] = sensemaker.memory
                results_dict[query_id]['run_1']["understanding"] = sensemaker.understanding
                results_dict[query_id]['run_1']["information_requests"] = sensemaker.information_request
                results_dict[query_id]['run_1']["function_calls"] = sensemaker.function_calls
                results_dict[query_id]['run_1']["step_history"] = sensemaker.step_history
                with open(name, "wb") as file:
                    pickle.dump(results_dict, file)
                print("Saving dictionary")


        except Exception as e:
            print("Error: ", e)
            with open(name, "wb") as file:
                pickle.dump(results_dict, file)
            print("Dictionary has been pickled and saved")


# run_subjective_evaluations_with_sensemaking(start=0, end=-1, name="gloss_subj_iter_1_run_1.pkl")

def delete_key(name, key_list):
    with open(name, "rb") as f:
        d = pickle.load(f)
        for key in key_list:
            if key in d:
                print(f"Deleting key: {key}")
                del d[key]

    with open(name, "wb") as f:
        pickle.dump(d, f)

# delete_key("gloss_obj_run_2.pkl", ["obj_query_5"])
# delete_key("gloss_obj_run_3.pkl", ["obj_query_6"])
# delete_key("gloss_obj_run_3.pkl", ["obj_query_7"])
# delete_key("gloss_obj_iter_1_run_1.pkl", ["obj_query_12"])

# generate_compliance_reports()