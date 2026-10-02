import pickle
import pandas as pd
import random


with open("rag_obj_iter_1_run_1.pkl", "rb") as f:
    results_obj_run1 = pickle.load(f)

with open("rag_obj_iter_1_run_2.pkl", "rb") as f:
    results_obj_run2 = pickle.load(f)

with open("rag_obj_iter_1_run_3.pkl", "rb") as f:
    results_obj_run3 = pickle.load(f)

df = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
q_dict = {}
for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
    q_dict[key] = query

query_id = []
queries = []
answers_run_1 = []
answers_run_2 = []
answers_run_3 = []
for key in q_dict.keys():
    query_id.append(key)
    queries.append(q_dict[key])
    if key in results_obj_run1:
        if "answer" in results_obj_run1[key]['run_1']:
            answers_run_1.append(results_obj_run1[key]['run_1']['answer'])
        else:
            answers_run_1.append("")
    else:
        answers_run_1.append("")
    if key in results_obj_run2:
        if "answer" in results_obj_run2[key]['run_1']:
            answers_run_2.append(results_obj_run2[key]['run_1']['answer'])
        else:
            answers_run_2.append("")
    else:
        answers_run_2.append("")
    if key in results_obj_run3:
        if "answer" in results_obj_run3[key]['run_1']:
            answers_run_3.append(results_obj_run3[key]['run_1']['answer'])
        else:
            answers_run_3.append("")

    else:
        answers_run_3.append("")
    # answers_run_2.append(results_obj_run2[key]['run_1']['answer'])
    # answers_run_3.append(results_obj_run3[key]['run_1']['answer'])

df = pd.DataFrame({'query_id': query_id, 'query': queries, 'answer_run_1': answers_run_1, 'answer_run_2': answers_run_2, 'answer_run_3': answers_run_3})
df.to_csv('rag_obj_consistency_iter_1.csv', index=False)