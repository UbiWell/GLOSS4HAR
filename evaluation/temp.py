import pickle
import pandas as pd
import random



def print_subj_answer():
    with open("gloss_obj_iter_1_run_1.pkl", "rb") as f:
        results_obj_run1 = pickle.load(f)

    df = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
    q_dict = {}
    for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
        q_dict[key] = query

    for key in results_obj_run1.keys():
        if "answer" not in results_obj_run1[key]['run_1']:
            print(key)
            print(q_dict[key])
            print(results_obj_run1[key]['run_1'])
            print("error for key", key)
            continue
        print(key)
        print(q_dict[key])
        print(f"Answer: {results_obj_run1[key]['run_1']['answer']}")

        # print(f"function: {results_obj_run1[key]['run_1']['function_calls']}")
        # print(f"memory: {results_obj_run1[key]['run_1']['memory']}")
        # print(results_obj_run1[key]['run_1']['memory'])
        print("\n\n")

    # query_ids = results_obj_run1.keys()
    # query_ids = list(df['query_id'])


def print_info_about(query_id, file_name):
    with open(file_name, "rb") as f:
        results_obj_run1 = pickle.load(f)

    df = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
    q_dict = {}
    for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
        q_dict[key] = query
    print(q_dict[query_id])
    print(f"Answer: {results_obj_run1[query_id]['run_1']['answer']}")
    print(f"function: {results_obj_run1[query_id]['run_1']['function_calls']}")
    print(f"memory: {results_obj_run1[query_id]['run_1']['memory']}")
    print(f"understanding: {results_obj_run1[query_id]['run_1']['understanding']}")
    print(f"information_requests: {results_obj_run1[query_id]['run_1']['information_requests']}")
    print(f"step_history: {results_obj_run1[query_id]['run_1']['step_history']}")


print_info_about("obj_query_40", "gloss_obj_iter_1_run_1.pkl")


import os

# Print all environment variables
# for key, value in os.environ.items():
#     print(f"{key}: {value}")


# print_subj_answer()