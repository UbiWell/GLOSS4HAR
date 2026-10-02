import random
import sys
import os
from matplotlib import ticker
import numpy as np
import pandas as pd
from bert_score import BERTScorer
from itertools import groupby
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import sklearn
import traceback
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))

import sensemaking_process
from datetime import datetime, timedelta
from data_streams.android_location import *
from data_streams.android_phone_usage import *
from data_streams.pixel_ambient_noises import *
from data_streams.pixel_hr import *
from data_streams.pixel_skin_temp import *
from data_streams.pixel_steps_data import *
from data_streams.pixel_uEMA import *
from data_streams.pixel_wear_detection import *

participants = {
    'pilot2': '2025/02/19', 'pilot5': '2025/02/25', 'pilot6': '2025/02/25',
    'pilot7': '2025/02/27', 'pilot8': '2025/02/27', 'pilot9': '2025/03/02',
    'pilot10': '2025/03/10', 'pilot11': '2025/03/03',
}

NO_PHONE_USAGE_TEXT = "No phone usage periods found in the specified time range."

BASE = "/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/"

ANNOTATION_PATHS = {
    "oss-gloss":              ("annotations_clean",      "{}_cleaned_results_oss.csv"),
    "gpt4-gloss":             ("annotations_clean",      "{}_cleaned_results.csv"),
    "oss-vanilla":            ("annotations_clean",      "{}_cleaned_results_vanilla_oss.csv"),
    "gpt4-vanilla":           ("annotations_clean",      "{}_cleaned_results_vanilla.csv"),
    "gpt4-uEMA":              ("annotation_task3",       "{}_cleaned_results_gpt4.csv"),
    "oss-uEMA":               ("annotation_task3",       "{}_cleaned_results.csv"),
    "gpt4-lst":               ("annotation_task4",       "{}_cleaned_results_gpt4.csv"),
    "oss-lst":                ("annotation_task4",       "{}_cleaned_results_oss.csv"),
    "gpt4-narrative":         ("annotation_task5",       "{}_cleaned_results_oss.csv"),
    "oss-narrative":          ("annotation_task5",       "{}_cleaned_results_gpt4.csv"),
    "gpt4-ps":                ("annotation_task2",       "{}_cleaned_results_gpt4.csv"),
    "oss-ps":                 ("annotation_task2",       "{}_cleaned_results_oss.csv"),
    "gpt4-sim":               ("annotation_sim",         "{}_cleaned_sim_gpt4.csv"),
    "oss-sim":                ("annotation_sim",         "{}_cleaned_sim_oss.csv"),
    "gpt4-lst-hard":          ("annotation_task6",       "{}_cleaned_results_gpt4.csv"),
    "oss-lst-hard":           ("annotation_task6",       "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-info":        ("ablation_information_loss",     "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-info":         ("ablation_information_loss",     "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-swapping":    ("ablation_swapping",      "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-swapping":     ("ablation_swapping",      "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-temporal-exact": ("ablation_temporal_exact", "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-temporal-exact":  ("ablation_temporal_exact", "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-temporal-20": ("ablation_temporal_20",   "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-temporal-20":  ("ablation_temporal_20",   "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-temporal-40": ("ablation_temporal_40",   "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-temporal-40":  ("ablation_temporal_40",   "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-temporal-60": ("ablation_temporal_60",   "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-temporal-60":  ("ablation_temporal_60",   "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-no-memory": ("annotations_clean_no_memory", "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-no-memory":  ("annotations_clean_no_memory", "{}_cleaned_results_oss.csv"),
    "gpt4-gloss-no-presentation": ("annotations_clean_no_presentation",   "{}_cleaned_results_gpt4.csv"),
    "oss-gloss-no-presentation":  ("annotations_clean_no_presentation",   "{}_cleaned_results_oss.csv"),
    "gpt4-ps-no-memory": ("annotation_task2_no_memory",   "{}_cleaned_results_gpt4.csv"),
    "oss-ps-no-memory":  ("annotation_task2_no_memory",   "{}_cleaned_results_oss.csv"),
    "gpt4-ps-no-presentation": ("annotation_task2_no_presentation",   "{}_cleaned_results_gpt4.csv"),
    "oss-ps-no-presentation":  ("annotation_task2_no_presentation",   "{}_cleaned_results_oss.csv"),
    "gpt4-lst-no-memory": ("annotation_task4_no_memory",   "{}_cleaned_results_gpt4.csv"),
    "oss-lst-no-memory":  ("annotation_task4_no_memory",   "{}_cleaned_results_oss.csv"),
    "gpt4-lst-no-presentation": ("annotation_task4_no_presentation",   "{}_cleaned_results_gpt4.csv"),
    "oss-lst-no-presentation":  ("annotation_task4_no_presentation",   "{}_cleaned_results_oss.csv"),
    "gpt4-lst-hard-no-memory": ("annotation_task6_no_memory",   "{}_cleaned_results_gpt4.csv"),
    "oss-lst-hard-no-memory":  ("annotation_task6_no_memory",   "{}_cleaned_results_oss.csv"),
    "gpt4-lst-hard-no-presentation": ("annotation_task6_no_presentation",   "{}_cleaned_results_gpt4.csv"),
    "oss-lst-hard-no-presentation":  ("annotation_task6_no_presentation",   "{}_cleaned_results_oss.csv"),
    "gpt4-uEMA-no-memory": ("annotation_task3_no_memory",   "{}_cleaned_results_gpt4.csv"),
    "oss-uEMA-no-memory":  ("annotation_task3_no_memory",   "{}_cleaned_results_oss.csv"),
    "gpt4-uEMA-no-presentation": ("annotation_task3_no_presentation",   "{}_cleaned_results_gpt4.csv"),
    "oss-uEMA-no-presentation":  ("annotation_task3_no_presentation",   "{}_cleaned_results_oss.csv"),
}

def get_all_results_from_participants(subject_id, a_type="oss-gloss"):
    subdir, tmpl = ANNOTATION_PATHS[a_type]
    path = os.path.join(BASE, subdir, tmpl.format(subject_id))
    try:
        df = pd.read_csv(path, sep=',', header=0, index_col=0)
        df = df.rename(columns={'uncertainty_start': 'start_time', 'uncertainty_end': 'end_time', 'labels': 'activity'})
        return df
    except Exception:
        return pd.DataFrame()

def parse_results_column(df, col_name):
    results = []
    for _, row in df.iterrows():
        try:
            segments = row[col_name].split('\n')
        except Exception:
            continue
        for seg in segments:
            try:
                time_part, rest = seg.split(': posture: ')
                time_part = time_part.replace('- ', '')
                posture_part, rest = rest.split('; activities: ')
                activity_part, reasoning_part = rest.split('; reasoning: ')
                start_time_str, end_time_str = time_part.split('-')
                results.append({
                    'start_time': start_time_str.strip(), 'end_time': end_time_str.strip(),
                    'posture': posture_part.strip(), 'activity': activity_part.strip(),
                    'reasoning': reasoning_part.strip()
                })
            except ValueError:
                continue
    return pd.DataFrame(results)

def read_annotated_file(subject_id):
    path = os.path.join(BASE, f"/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2/{subject_id}_cleaned.csv")
    df = pd.read_csv(path, sep=',', header=0, index_col=0)
    return df[['date', 'uncertainty_start', 'uncertainty_end', 'labels']].rename(
        columns={'uncertainty_start': 'start_time', 'uncertainty_end': 'end_time', 'labels': 'activity'})

def split_into_minute_level(df):
    rows = []
    for _, row in df.iterrows():
        try:
            act = row['activity']
            if 'posture' in row:
                act = row['posture'] + ', ' + act
            t = datetime.strptime(row['start_time'], "%H:%M")
            end = datetime.strptime(row['end_time'], "%H:%M")
            while t < end:
                rows.append({'timestamp': t.strftime("%H:%M"), 'activity': act})
                t += timedelta(minutes=1)
        except Exception:
            continue
    return pd.DataFrame(rows)

def clean_activity_labels(df, col):
    replacements = {
        'conversing': 'talking', 'body conditioning': 'strength training',
        'walking (brisk)': 'walking', 'running': 'walking',
        'shopping (non-food)': 'shopping', 'shopping (food)': 'shopping',
        'get dressed': 'getting dressed', 'watching contents': 'watching movies/TV',
        'dusting furniture': 'cleaning', 'packing boxes': 'packing/unpacking',
        'other activities': '',
    }
    for old, new in replacements.items():
        df[col] = df[col].str.replace(old, new, regex=False)
    skip = {'none', 'null', 'unknown', 'other activities', ''}
    df[col] = df[col].apply(lambda x: ', '.join(sorted(set(a.strip() for a in x.split(',') if a.strip().lower() not in skip))))
    return df

def get_all_activities():
    return list(set([
        'video gaming','walking','stair climbing','getting ready','driving','bicycling',
        'vigorous bicycling','aerobics','cleaning','cooking','laundry','playing with pet',
        'listening to music','watching movies/TV','studying','reading','riding car',
        'riding train','riding bus','playing musical instruments','attending meeting',
        'computer using','phone using','running','getting dressed','grooming',
        'using bathroom','eating','talking','strength training','washing dishes',
        'carrying groceries','putting away groceries','shopping','making bed',
        'packing/unpacking','sleeping','playing sports','sitting','standing',
        'lying down','reclining','upright','crouching','kneeling','using phone',
    ]))

def process_multilabel(dfs):
    all_activities = get_all_activities()
    matrices = []
    for df in dfs:
        col = df.columns[0]
        mat = pd.DataFrame(0, index=df.index, columns=sorted(all_activities))
        for idx, acts in df[col].items():
            for act in (a.strip() for a in acts.split(',')):
                if act in mat.columns: mat.at[idx, act] = 1
        matrices.append(mat)
    return matrices

def exact_match_accuracy(y_true, y_pred): return sklearn.metrics.accuracy_score(y_true, y_pred)
def f1_score(y_true, y_pred):             return sklearn.metrics.f1_score(y_true, y_pred, average='micro')
def hamming_loss(y_true, y_pred):         return sklearn.metrics.hamming_loss(y_true, y_pred)
def jaccard_index(y_true, y_pred):        return sklearn.metrics.jaccard_score(y_true, y_pred, average='samples')

def get_objective_locomotion_ground_truth(subject_id):
    df = read_annotated_file(subject_id)
    df = clean_activity_labels(df, 'activity')
    df = split_into_minute_level(df)

    date = participants[subject_id].replace('/', '-')
    start_time = f"{date} {df.iloc[0]['timestamp']}:00"
    end_time = f"{date} {df.iloc[-1]['timestamp']}:59"

    step_df = pd.DataFrame(get_steps_records(subject_id, start_time, end_time))
    step_df['locomotion'] = step_df['steps'].apply(lambda x: 'walking' if x > 10 else '')
    step_df['timestamp'] = step_df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%H:%M'))
    df = pd.merge(df, step_df[['timestamp', 'locomotion']], on='timestamp', how='left')
    df['locomotion'] = df['locomotion'].fillna('')

    phone_usage_df = pd.DataFrame(get_phone_usage_records(subject_id, start_time, end_time))
    if phone_usage_df.empty:
        phone_usage_df = pd.DataFrame(columns=['timestamp', 'in_use'])
    phone_usage_df['locomotion'] = phone_usage_df['in_use'].apply(lambda x: 'using phone' if x == 'True' else '')
    phone_usage_df['timestamp'] = phone_usage_df['timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%H:%M'))
    df = pd.merge(df, phone_usage_df[['timestamp', 'locomotion']], on='timestamp', how='left', suffixes=('', '_phone'))

    for idx, row in df.iterrows():
        if row['locomotion'] == '':
            acts = [a.strip() for a in row['activity'].split(',')
                    if a.strip().lower() not in ('using phone', 'walking', 'running')]
            df.at[idx, 'locomotion'] = ', '.join(acts)
        else:
            df.at[idx, 'locomotion'] = row['locomotion']

        if row['locomotion_phone'] == 'phone using' and 'phone using' not in df.at[idx, 'locomotion']:
            df.at[idx, 'locomotion'] = (df.at[idx, 'locomotion'] + ', phone using').lstrip(', ')
        elif row['locomotion_phone'] == '' and 'phone using' in df.at[idx, 'locomotion']:
            acts = [a.strip() for a in df.at[idx, 'locomotion'].split(',') if a.strip().lower() != 'phone using']
            df.at[idx, 'locomotion'] = ', '.join(acts)

    return df.drop(columns=['locomotion_phone'])

def run_multilabel_metrics(subject_id, a_type="gpt4-gloss", ground_truth="thigh_sensor"):
    result_df = read_annotated_file(subject_id) if a_type == "original" else get_all_results_from_participants(subject_id, a_type=a_type)
    if a_type != "original":
        result_df = parse_results_column(result_df, 'result')
    try:
        result_df = clean_activity_labels(result_df, 'activity')
    except Exception:
        return None
    result_df = split_into_minute_level(result_df)

    if ground_truth == "annotated_locomotion":
        gt_df = clean_activity_labels(read_annotated_file(subject_id), 'activity')
        gt_df = split_into_minute_level(gt_df).rename(columns={'activity': 'locomotion'})
    else:
        gt_df = get_objective_locomotion_ground_truth(subject_id)
    gt_df = gt_df[['timestamp', 'locomotion']]

    result_df = result_df.groupby('timestamp').agg({'activity': lambda x: ', '.join(x)}).reset_index()
    result_df['activity'] = result_df['activity'].apply(lambda x: ', '.join(sorted(set(i.strip() for i in x.split(',')))))

    if ground_truth == "thigh_sensor":
        kept = {"sitting","standing","lying down","walking","crouching","kneeling"}
        result_df['activity'] = result_df['activity'].apply(lambda x: ', '.join(sorted(set(i.strip() for i in x.split(', ') if i.strip() in kept))))
        gt_df['locomotion']   = gt_df['locomotion'].apply(lambda x: ', '.join(sorted(set(i.strip() for i in x.split(', ') if i.strip() in kept))))

    merged = pd.merge(gt_df, result_df, on='timestamp', how='inner')
    if a_type != "original":
        merged['activity'] = merged.apply(lambda r: r['activity'] + ', sitting' if 'upright' in r['activity'] else r['activity'], axis=1)
        merged['activity'] = merged.apply(lambda r: r['activity'] + ', standing' if 'upright' in r['activity'] else r['activity'], axis=1)
        merged['activity'] = merged.apply(
            lambda r: ', '.join(sorted(set(i.strip() for i in r['activity'].split(', ') if i.strip() in r['locomotion'].split(', ')))), axis=1)
    merged = merged[(merged['locomotion'] != '') & (merged['activity'] != '')]

    try:
        os.makedirs(os.path.join(BASE, 'fixed'), exist_ok=True)
        merged.to_csv(os.path.join(BASE, f'fixed/{subject_id}_{a_type}_merged.csv'))
    except Exception as e:
        print(f"Error saving merged df for subject {subject_id}: {e}")

    return process_multilabel([merged[['locomotion']], merged[['activity']]])


# ── t-test logic ──────────────────────────────────────────────────────────────

METRICS = ["exact_match", "f1", "hamming", "jaccard"]

def get_per_subject_metrics(a_type, ground_truth="annotated_locomotion"):
    results = {}
    for subject in participants:
        try:
            result = run_multilabel_metrics(subject, a_type=a_type, ground_truth=ground_truth)
            if result is None:
                continue
            y_true, y_pred = result
            if y_true.shape[0] == 0:
                continue
            results[subject] = {
                "exact_match": exact_match_accuracy(y_true, y_pred),
                "f1":          f1_score(y_true, y_pred),
                "hamming":     hamming_loss(y_true, y_pred),
                "jaccard":     jaccard_index(y_true, y_pred),
            }
        except Exception as e:
            print(f"  [WARN] {a_type} / {subject}: {e}")
    return results

def run_paired_ttest(scores_a, scores_b, metric):
    common = sorted(set(scores_a) & set(scores_b))
    if len(common) < 2:
        return None
    a = np.array([scores_a[s][metric] for s in common])
    b = np.array([scores_b[s][metric] for s in common])
    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "n":             len(common),
        "mean_baseline": np.mean(a),
        "mean_other":    np.mean(b),
        "diff":          np.mean(b) - np.mean(a),
        "t":             t_stat,
        "p":             p_val,
        "sig":           "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")),
    }

def main():
    ground_truth = "annotated_locomotion"
    models       = ["gpt4", "oss"]
    baseline     = "ps"
    comparisons  = ["uEMA", "lst", "lst-hard"]

    conditions_needed = set()
    for model in models:
        conditions_needed.add(f"{model}-{baseline}")
        for comp in comparisons:
            conditions_needed.add(f"{model}-{comp}")

    print("Collecting per-subject metrics for all conditions...")
    all_scores = {}
    for cond in sorted(conditions_needed):
        print(f"  {cond}...")
        all_scores[cond] = get_per_subject_metrics(cond, ground_truth)

    print("\n" + "="*80)
    print("PAIRED T-TEST RESULTS  (baseline = -ps, ground truth = annotated_locomotion)")
    print("="*80)

    rows = []
    for model in models:
        base_key = f"{model}-{baseline}"
        for comp in comparisons:
            comp_key = f"{model}-{comp}"
            for metric in METRICS:
                res = run_paired_ttest(all_scores[base_key], all_scores[comp_key], metric)
                if res is None:
                    print(f"[SKIP] {base_key} vs {comp_key} / {metric}: not enough subjects")
                    continue
                rows.append({
                    "model":      model,
                    "baseline":   base_key,
                    "comparison": comp_key,
                    "metric":     metric,
                    **res,
                })

    df = pd.DataFrame(rows)

    for model in models:
        print(f"\n{'─'*80}")
        print(f"  MODEL: {model.upper()}")
        print(f"{'─'*80}")
        sub = df[df["model"] == model]
        for comp in comparisons:
            comp_key = f"{model}-{comp}"
            print(f"\n  {model}-ps  vs  {comp_key}")
            print(f"  {'Metric':<14} {'M(ps)':>8} {'M(other)':>10} {'Diff':>8} {'t':>8} {'p':>10} {'sig':>5}")
            print(f"  {'-'*64}")
            for _, row in sub[sub["comparison"] == comp_key].iterrows():
                arrow = "↑" if row["diff"] > 0 else "↓"
                print(f"  {row['metric']:<14} {row['mean_baseline']:>8.4f} {row['mean_other']:>10.4f} "
                      f"{row['diff']:>+8.4f}{arrow} {row['t']:>8.3f} {row['p']:>10.4f} {row['sig']:>5}")

    out = os.path.join(os.path.dirname(__file__), "ttest_results.csv")
    df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")

if __name__ == "__main__":
    main()