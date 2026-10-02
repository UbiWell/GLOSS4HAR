import os
import pickle
from agents.rag_based_agent import RAGBasedAgent
import pandas as pd
from agents.gpt_utils import invoke_with_retry


# def clear_dict(name):
#     with open(name, "wb") as file:
#         pickle.dump({}, file)

def run_objective_evaluations_with_sensemaking(start = 0, end = -1, name="results.pkl", rerun=False):
    df = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
    # df = df[df['labels'] == 'objective' or df['labels'] == 'objective+']
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
        for query_id, query in zip(query_id_to_run, queries_to_run):
            print(f"Query: {query}")
            print(f"Running query: {query_id}")
            # if "stress" in query:
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
            try:
                response = invoke_with_retry(RAGBasedAgent(), "invoke_rag_agent", {'user_query': query})
                if response == "FAILED":
                    print("Failed to get response")
                    continue
            # sys.stdout = sys.__stdout__
                results_dict[query_id]['run_1']["answer"] = response

                with open(name, "wb") as file:
                    pickle.dump(results_dict, file)
                print("Saving dictionary")
            except Exception as e:
                print("Error: ", e)
                print(f"skipping {query_id} due to error")
                continue


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
        for query_id, query in zip(query_id_to_run, queries_to_run):
            print(f"Query: {query}")
            print(f"Running query: {query_id}")
            # if "stress" in query:
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
            try:
                response = invoke_with_retry(RAGBasedAgent(), "invoke_rag_agent", {'user_query': query})
                if response == "FAILED":
                    print("Failed to get response")
                    continue
            # sys.stdout = sys.__stdout__
                results_dict[query_id]['run_1']["answer"] = response

                with open(name, "wb") as file:
                    pickle.dump(results_dict, file)
                print("Saving dictionary")
            except Exception as e:
                print("Error: ", e)
                print(f"skipping {query_id} due to error")
                continue


# clear_dict("rag_results_subj_run_1.pkl")
# run_subjective_evaluations_with_sensemaking(start=0, end=-1, name="rag_subj_iter_1_run_1.pkl", rerun=False)


run_objective_evaluations_with_sensemaking(start=0, end=-1, name=f"rag_obj_iter_1_run_3.pkl", rerun=False)

# # response = RAGBasedAgent().invoke_rag_agent({'user_query': "query"})
# print(response)