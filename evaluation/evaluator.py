import pickle
import pandas as pd
import random
from scipy.stats import ttest_rel
import numpy as np


# with open("results_obj_run1.pkl", "rb") as f:
#     results_obj_run1 = pickle.load(f)
#
# df = pd.read_csv("objective_human_queries_with_labels.csv")
# q_dict = {}
# for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
#     q_dict[key] = query
#
# # for key in results_obj_run1.keys():
# #     if "answer" not in results_obj_run1[key]['run_1']:
# #         print("error for key", key)
# #         continue
# #     print(key)
# #     print(q_dict[key])
# #     print(f"Answer: {results_obj_run1[key]['run_1']['answer']}")
# #     # print(results_obj_run1[key]['run_1']['memory'])
# #     print("\n\n")
#
# query_ids = results_obj_run1.keys()
# query_ids = list(df['query_id'])
# sampled_queries = random.sample(query_ids, 30)

# queries = []
# for s in sampled_queries:
#     queries.append(q_dict[s])
#
# df_sample = pd.DataFrame()
# df_sample['query'] = queries
# df_sample['query_id'] = sampled_queries
# df_sample['query_id_num'] = df_sample['query_id'].str.extract(r'(\d+)', expand=False).astype(int)
# df_sample = df_sample.sort_values(by='query_id_num').drop(columns=['query_id_num'])
# print ("sampled queries")
# df_sample.to_csv("sampled_queries_for_accuracy.csv", index=False)

def common_words(sentence1, sentence2):
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())
    common = set1 & set2
    return len(common), common
def get_corresponding_query():
    df = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
    q_dict = {}
    for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
        q_dict[key] = query

    query_ids = list(df['query_id'])
    sampled_queries = random.sample(query_ids, 32)

    queries = []
    for s in sampled_queries:
        queries.append(q_dict[s])

    df_sample = pd.DataFrame()
    df_sample['query'] = queries
    df_sample['query_id'] = sampled_queries
    df_sample['query_id_num'] = df_sample['query_id'].str.extract(r'(\d+)', expand=False).astype(int)
    df_sample = df_sample.sort_values(by='query_id_num').drop(columns=['query_id_num'])
    print ("sampled queries")
    df_sample.to_csv("sampled_queries_for_accuracy.csv", index=False)

    df_sample = pd.read_csv("sampled_queries_for_accuracy.csv")
    query_ids = list(df_sample['query_id'])
    queries = list(df_sample['query'])
    added_queries = []
    added_query_ids = []

    for q in query_ids:
        q_index  = int(q.split("_")[2])
        if(q_index == 1):
            added_queries.append(q_dict[f"obj_query_{q_index + 1}"])
            added_query_ids.append(f"obj_query_{q_index + 1}")
            continue
        if(q_index == len(query_ids)):
            added_queries.append(q_dict[f"obj_query_{q_index - 1}"])
            added_query_ids.append(f"obj_query_{q_index - 1}")
            continue
        left = f"obj_query_{q_index - 1}"
        right = f"obj_query_{q_index + 1}"
        query = q_dict[q]
        left_query = q_dict[left]
        right_query = q_dict[right]
        c1,_ = common_words(query, left_query)
        c2,_ = common_words(query, right_query)
        if c1 > c2:
            added_queries.append(left_query)
            added_query_ids.append(left)
        else:
            added_queries.append(right_query)
            added_query_ids.append(right)

    total_queries = queries + added_queries
    total_query_ids = query_ids + added_query_ids
    df_sample_new = pd.DataFrame()
    df_sample_new['query'] = total_queries
    df_sample_new['query_id'] = total_query_ids
    df_sample_new['query_id_num'] = df_sample_new['query_id'].str.extract(r'(\d+)', expand=False).astype(int)
    df_sample_new = df_sample_new.sort_values(by='query_id_num').drop(columns=['query_id_num'])
    print(df_sample_new)
    print(len(set(df_sample_new['query_id'])))
    df_sample_new.drop_duplicates()
    df_sample_new.to_csv("sampled_queries_for_accuracy_added.csv", index=False)

#
# get_corresponding_query()

def get_sample_queries():
    df = pd.read_csv("sampled_queries_for_accuracy_added.csv")
    ids = list(df['query_id'])
    query_ids = []
    queries = []

    df_obj = pd.read_csv("objective_human_queries_with_labels_iter_1.csv")
    for i, row in df_obj.iterrows():
        if row['query_id'] in ids:
            query_ids.append(row['query_id'])
            queries.append(row['updated_queries'])

    df_sample = pd.DataFrame()
    df_sample['query_id'] = query_ids
    df_sample['query'] = queries
    df_sample.to_csv("sampled_queries_for_accuracy_added_iter_1.csv", index=False)


def get_sample_answer():
    df = pd.read_csv("sampled_queries_for_accuracy_added_iter_1.csv")
    file_name = "rag_obj_iter_1_run_1.pkl"
    with open(file_name, "rb") as f:
        results_obj_run1 = pickle.load(f)

    q_ids = df['query_id']
    queries = df['query']
    answers = []
    understanding = []
    action_plan = []
    memory = []
    function_calls = []
    steps = []
    information_requests = []
    for q_id in q_ids:
        ans = results_obj_run1[q_id]['run_1']['answer']
        answers.append(ans)
        # understanding.append(results_obj_run1[q_id]['run_1']['understanding'])
        # action_plan.append(results_obj_run1[q_id]['run_1']['action_plan'])
        # memory.append(results_obj_run1[q_id]['run_1']['memory'])
        # function_calls.append(results_obj_run1[q_id]['run_1']['function_calls'])
        # information_requests.append(results_obj_run1[q_id]['run_1']['information_requests'])
        # steps.append(results_obj_run1[q_id]['run_1']['step_history'])

    df_new = pd.DataFrame()
    df_new['query_id'] = q_ids
    df_new['query'] = queries
    df_new['answer'] = answers
    # df_new['action_plan'] = action_plan
    # df_new['information_requests'] = information_requests
    # df_new['steps'] = steps
    # df_new['understanding'] = understanding
    # df_new['memory'] = memory
    # df_new['function_calls'] = function_calls

    df_new.to_csv("sampled_queries_for_accuracy_added_iter_1_run_1_with_answer_rag.csv", index=False)









def calc_accuracy():
    df = pd.read_csv("sampled_queries_for_accuracy_added_iter_1.csv")
    correct = 0.0
    total = 0
    incorrect = 0.0
    # print(df.columns)
    for i, row in df.iterrows():
        if row['correct_iter_1_run_1'] == "unsure":
            continue
        total += 1
        if row['correct_iter_1_run_1'] == "yes":
            correct += 1
        elif row['correct_iter_1_run_1'] == "no":
            incorrect += 1
        else:
            print(row['correct_iter_1_run_1'])
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Total: {total}")
    print(f"Accuracy: {(correct/(correct + incorrect))*100}")


def calc_accuracy_rag():
    df = pd.read_csv("sampled_queries_for_accuracy_added_iter_1.csv")
    correct = 0.0
    total = 0
    incorrect = 0.0
    # print(df.columns)
    for i, row in df.iterrows():
        if row['correct_rag'] == "unsure":
            continue
        total += 1
        if row['correct_rag'] == "yes":
            correct += 1
        elif row['correct_rag'] == "no":
            incorrect += 1
        else:
            print(row['correct_rag'])
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Total: {total}")
    print(f"Accuracy: {(correct/(correct + incorrect))*100}")

# calc_accuracy()

def generate_subjective_evaluation_file():
    with open("rag_subj_iter_1_run_1.pkl", "rb") as f:
        rag_subj = pickle.load(f)

    with open("gloss_subj_iter_1_run_1.pkl", "rb") as f:
        gloss_subj = pickle.load(f)

    df = pd.read_csv("subjective_human_queries_with_labels_iter_1.csv")
    q_dict = {}
    for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
        q_dict[key] = query

    q_ids = []
    queries = []
    answers = []
    model = []

    for k in q_dict:
        rag_answer = rag_subj[k]['run_1']['answer']
        gloss_answer = gloss_subj[k]['run_1']['answer']
        q_ids.append(k)
        queries.append(q_dict[k])
        answers.append(rag_answer)
        model.append("rag")
        q_ids.append(k)
        queries.append(q_dict[k])
        answers.append(gloss_answer)
        model.append("gloss")

    df_new = pd.DataFrame()
    df_new['query_id'] = q_ids
    df_new['query'] = queries
    df_new['answer'] = answers
    df_new['model'] = model
    #shuffle df
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new.to_csv("subjective_evaluation_file_iter_1_run_1.csv", index=False)

generate_subjective_evaluation_file()


def generate_subjective_choose_one():
    with open("rag_subj_iter_1_run_1.pkl", "rb") as f:
        rag_subj = pickle.load(f)

    with open("gloss_subj_iter_1_run_1.pkl", "rb") as f:
        gloss_subj = pickle.load(f)

    df = pd.read_csv("subjective_human_queries_with_labels_iter_1.csv")
    q_dict = {}
    for key, query in zip(list(df['query_id']), list(df['updated_queries'])):
        q_dict[key] = query

    q_ids = []
    queries = []
    answer_1 = []
    answer_2 = []
    model = []
    choice = []

    for k in q_dict:
        rag_answer = rag_subj[k]['run_1']['answer']
        gloss_answer = gloss_subj[k]['run_1']['answer']
        q_ids.append(k)
        queries.append(q_dict[k])
        gloss_random_choice = random.choice([1,2])
        if gloss_random_choice == 1:
            answer_1.append(gloss_answer)
            answer_2.append(rag_answer)
        else:
            answer_1.append(rag_answer)
            answer_2.append(gloss_answer)
        choice.append(gloss_random_choice)


    df_new = pd.DataFrame()
    df_new['query_id'] = q_ids
    df_new['query'] = queries
    df_new['answer_1'] = answer_1
    df_new['answer_2'] = answer_2
    df_new['choice'] = choice

    #shuffle df
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    df_new.to_csv("subjective_evaluation_choose_one_iter_1_run_1.csv", index=False)

def calc_consistency():
    df = pd.read_csv("gloss_obj_consistency_iter_1.csv")
    correct = 0.0
    total = 0
    incorrect = 0.0
    # print(df.columns)
    for i, row in df.iterrows():
        if row['consistency'] == "unsure":
            continue
        total += 1
        if row['consistency'] == "three":
            correct += 1
        else:
            incorrect += 1

    print ("GLOSS")
    print(f"Incorrect: {incorrect}")
    print(f"Total: {total}")
    print(f"Consistency: {(correct/(correct + incorrect))*100}")

    df = pd.read_csv("rag_obj_consistency_iter_1.csv")
    correct = 0.0
    total = 0
    incorrect = 0.0
    for i, row in df.iterrows():
        if row['consistency'] == "unsure":
            continue
        total += 1
        if row['consistency'] == "three":
            correct += 1
        else:
            incorrect += 1
    print("RAG")
    print(f"Incorrect: {incorrect}")
    print(f"Total: {total}")
    print(f"Consistency: {(correct/(correct + incorrect))*100}")



def calc_significance_accuracy():
    df = pd.read_csv("gloss_obj_consistency_iter_1.csv")

    print(df.columns)
    # Get correct answers for each model
    correct_A = (df['correct_iter_1_run_1_gloss'] == 'yes').astype(int)
    correct_B = (df['correct_rag'] == 'yes').astype(int)

    # Perform paired t-test
    t_stat, p_value = ttest_rel(correct_A, correct_B)

    print(f'T-statistic: {t_stat:.4f}, P-value: {p_value}')

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("Statistically significant difference between model accuracies!")
    else:
        print("No significant difference between model accuracies.")


def calc_significance_consistency():
    df1 = pd.read_csv("gloss_obj_consistency_iter_1.csv")
    df2 = pd.read_csv("rag_obj_consistency_iter_1.csv")


    print(df1.columns)
    # Get correct answers for each model
    correct_A = (df1['consistency'] == 'three').astype(int)
    correct_B = (df2['consistency'] == 'three').astype(int)

    # Perform paired t-test
    print(correct_B)
    t_stat, p_value = ttest_rel(correct_A, correct_B)

    print(f'T-statistic: {t_stat:.4f}, P-value: {p_value}')

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("Statistically significant difference between model accuracies!")
    else:
        print("No significant difference between model consistency.")

# calc_accuracy_rag()
#
# get_sample_answer()

# calc_consistency()

calc_significance_consistency()