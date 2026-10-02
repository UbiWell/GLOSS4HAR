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

# Maps a_type to (subdir, filename_template)
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
    "oss-uEMA-no-presentation":  ("annotation_task3_no_presentation",   "{}_cleaned_results_oss.csv")
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

def parse_participants_original_annotations(subject_id):
    path = os.path.join(BASE, "annotations_clean", f"{subject_id}_cleaned_results.csv")
    df = pd.read_csv(path, sep=',', header=0, index_col=0)
    df = df[['uncertainty_start', 'uncertainty_end', 'labels']].rename(
        columns={'uncertainty_start': 'start_time', 'uncertainty_end': 'end_time', 'labels': 'activity'})
    df['posture'] = ''
    df['reasoning'] = ''
    return df

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

def objective_locomotion_check(subject_id, df=None, type="gpt4-gloss"):
    if df is None:
        if type == "gpt4-gloss":       results_df = parse_results_column(pd.read_csv(os.path.join(BASE, "annotations_clean", f"{subject_id}_cleaned_results.csv"), sep=',', header=0, index_col=0)[['result']], 'result')
        elif type == "original":        results_df = parse_participants_original_annotations(subject_id)
        elif type == "gpt4-vanilla":    results_df = parse_results_column(pd.read_csv(os.path.join(BASE, "annotations_clean", f"{subject_id}_cleaned_results_vanilla.csv"), sep=',', header=0, index_col=0)[['result']], 'result')
        elif type == "oss-vanilla":     results_df = parse_results_column(pd.read_csv(os.path.join(BASE, "annotations_clean", f"{subject_id}_cleaned_results_vanilla_oss.csv"), sep=',', header=0, index_col=0)[['result']], 'result')
        elif type == "oss-gloss":       results_df = parse_results_column(pd.read_csv(os.path.join(BASE, "annotations_clean", f"{subject_id}_cleaned_results_oss.csv"), sep=',', header=0, index_col=0)[['result']], 'result')
    else:
        results_df = df

    incorrect_minutes = total_minutes = 0
    date_str = participants[subject_id]

    for _, row in results_df.iterrows():
        try:
            start_time = datetime.strptime(f"{date_str} {row['start_time']}", "%Y/%m/%d %H:%M")
            end_time   = datetime.strptime(f"{date_str} {row['end_time']}",   "%Y/%m/%d %H:%M")
            start_str, end_str = start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')
            duration = (end_time - start_time).total_seconds() / 60
            activity, posture = row['activity'], row['posture']

            if duration > 300:
                continue

            steps_records = get_steps_records(subject_id, start_str, end_str)
            if any(kw in activity.lower() or kw in posture.lower() for kw in ('walking', 'running')):
                incorrect_minutes += sum(1 for r in steps_records if r['steps'] == 0)
            else:
                incorrect_minutes += sum(1 for r in steps_records if r['steps'] > 0)
            total_minutes += duration

            phone_records = get_phone_usage_records(subject_id, start_str, end_str)
            if 'using phone' in activity.lower():
                total_minutes += duration
                if get_phone_usage_period(subject_id, start_str, end_str) == NO_PHONE_USAGE_TEXT:
                    incorrect_minutes += total_minutes
            else:
                total_minutes += duration
                if phone_records:
                    incorrect_minutes += sum(1 for r in phone_records if r['in_use'] == 'True')
        except Exception:
            continue

    accuracy = (1 - incorrect_minutes / total_minutes) * 100 if total_minutes > 0 else 0
    return incorrect_minutes, total_minutes, accuracy

def _run_for_everyone(fn, label=""):
    total_incorrect = total_mins = 0
    for subject in participants:
        inc, mins, acc = fn(subject)
        print(f"Subject: {subject}, Incorrect: {inc}, Total: {mins}, Accuracy: {acc:.2f}%")
        total_incorrect += inc; total_mins += mins
    overall = (1 - total_incorrect / total_mins) * 100 if total_mins > 0 else 0
    print(f"Overall Accuracy: {overall:.2f}%")

def run_objective_check_for_everyone(type="gpt4-gloss"):
    _run_for_everyone(lambda s: objective_locomotion_check(s, type=type))

def run_objective_check_for_everyone_using_df(type="gpt4-gloss"):
    total_incorrect = total_mins = 0
    for subject in participants:
        df = get_all_results_from_participants(subject, a_type=type)
        if df.empty: continue
        df = parse_results_column(df, 'result')
        inc, mins, acc = objective_locomotion_check(subject, df=df, type=type)
        print(f"Subject: {subject}, Incorrect: {inc}, Total: {mins}, Accuracy: {acc:.2f}%")
        total_incorrect += inc; total_mins += mins
    overall = (1 - total_incorrect / total_mins) * 100 if total_mins > 0 else 0
    print(f"Overall Accuracy: {overall:.2f}%")

def read_annotated_file(subject_id):
    # f"/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotation_task2/{subject_id}_cleaned.csv"
    path = os.path.join(BASE, f"/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/{subject_id}_cleaned_results_vanilla.csv")
    df = pd.read_csv(path, sep=',', header=0, index_col=0)
    return df[['date', 'uncertainty_start', 'uncertainty_end', 'labels']].rename(
        columns={'uncertainty_start': 'start_time', 'uncertainty_end': 'end_time', 'labels': 'activity'})

def read_fixed_annotated_file(subject_id):
    path = os.path.join(BASE, f"/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations_clean/{subject_id}_cleaned_results_vanilla.csv")
    df = pd.read_csv(path, sep=',', header=0, index_col=0)
    return parse_results_column(df, 'result')

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

def remove_none_activities(df):
    skip = {'none', 'null', 'unknown', 'other activities', ''}
    for col in ('annotated_activity', 'result_activity'):
        df[col] = df[col].apply(lambda x: ', '.join(a.strip() for a in x.split(',') if a.strip().lower() not in skip))
    df['annotated_activity'] = df['annotated_activity'].str.replace('upright', 'standing', regex=False)
    df['result_activity'] = df.apply(
        lambda r: r['result_activity'].replace('upright', 'sitting' if 'sitting' in r['annotated_activity'] else 'standing'), axis=1)
    for col in ('annotated_activity', 'result_activity'):
        df[col] = df[col].apply(lambda x: ', '.join(sorted(set(a.strip() for a in x.split(',')))))
    return df

def mapping_timeline(results_df, annotated_df):
    ann = split_into_minute_level(annotated_df).rename(columns={'activity': 'annotated_activity'})
    res = split_into_minute_level(results_df).rename(columns={'activity': 'result_activity'})
    merged = pd.merge(ann, res, on='timestamp', how='outer').sort_values('timestamp')
    merged = merged.dropna(subset=['annotated_activity']).dropna(subset=['result_activity'])
    clean_activity_labels(merged, 'annotated_activity')
    clean_activity_labels(merged, 'result_activity')
    return merged.reset_index(drop=True)

def count_errors_in_mapping(mapped_df):
    skip = {'', 'other activities', 'none', 'null', 'unknown'}
    errors = total = 0
    for _, row in mapped_df.iterrows():
        acts = row['annotated_activity'].split(', ')
        total += len(acts) if acts else 1
        for act in acts:
            if act.strip().lower() in skip: continue
            if act not in row['result_activity']:
                if act in ('sitting', 'standing') and 'upright' in row['result_activity']: continue
                errors += 1
    return errors, total, (1 - errors / total) * 100 if total > 0 else 0

def run_timeline_mapping_for_everyone(type="gpt4-gloss"):
    total_total = total_errors = 0
    acc_list = []
    for subject in participants:
        df = get_all_results_from_participants(subject, a_type=type)
        if df.empty: continue
        df = parse_results_column(df, 'result')
        ann = read_fixed_annotated_file(subject)
        mapped = mapping_timeline(df, ann)
        errs, total, acc = count_errors_in_mapping(mapped)
        total_total += total; total_errors += errs; acc_list.append(acc)
    print(f"Overall Accuracy: {(1 - total_errors / total_total) * 100:.2f}%; M = {np.mean(acc_list):.2f}%, SD = {np.std(acc_list):.2f}%")

def misalignment_analysis(type="gpt4-gloss"):
    totals = dict(insertion=0, deletion=0, over_underfill=0, fragment_merge=0, true_positives=0, total_gt_minutes=0)
    for subject in participants:
        df = get_all_results_from_participants(subject, a_type=type)
        if df.empty: continue
        df = parse_results_column(df, 'result')
        ann = read_fixed_annotated_file(subject)
        mapped = mapping_timeline(df, ann)
        res = calculate_ar_errors_atomic(mapped)
        for k in totals: totals[k] += res[k]
    gt = totals['total_gt_minutes']
    for key in ('insertion', 'deletion', 'over_underfill', 'fragment_merge'):
        label = key.replace('_', '/').title()
        print(f"Overall {label}: {totals[key]} ({totals[key] / gt * 100:.2f}% / {gt})")

def calculate_ar_errors_atomic(df, gt_col="annotated_activity", pred_col="result_activity"):
    df['annotated_set'] = df[gt_col].apply(lambda x: set(x.split(', ')))
    df['result_set']    = df[pred_col].apply(lambda x: set(x.split(', ')))

    tp_counter = Counter()
    for _, row in df.iterrows():
        for act in row['annotated_set'] & row['result_set']:
            tp_counter[act] += 1

    def get_contiguous_events(series):
        events = {}
        for act in set().union(*series):
            indices = [i for i, s in enumerate(series) if act in s]
            blocks = []
            if indices:
                s = e = indices[0]
                for i in indices[1:]:
                    if i == e + 1: e = i
                    else: blocks.append((s, e)); s = e = i
                blocks.append((s, e))
            events[act] = blocks
        return events

    gt_events   = get_contiguous_events(df['annotated_set'])
    pred_events = get_contiguous_events(df['result_set'])

    total_gt_minutes = sum(e - s + 1 for blocks in gt_events.values() for s, e in blocks)

    overfill_c = underfill_c = fragmentation_c = merge_c = Counter(), Counter(), Counter(), Counter()
    overfill_c, underfill_c, fragmentation_c, merge_c = Counter(), Counter(), Counter(), Counter()

    def overlaps(a_s, a_e, b_s, b_e): return not (a_e < b_s or a_s > b_e)

    for act, gt_blocks in gt_events.items():
        p_blocks = pred_events.get(act, [])
        for gs, ge in gt_blocks:
            ov = [(ps, pe) for ps, pe in p_blocks if overlaps(gs, ge, ps, pe)]
            if ov:
                ep = min(ps for ps, _ in ov); lp = max(pe for _, pe in ov)
                underfill_c[act] += max(0, ep - gs) + max(0, ge - lp)
                if len(ov) > 1: fragmentation_c[act] += len(ov) - 1

    for act, p_blocks in pred_events.items():
        gt_blocks = gt_events.get(act, [])
        for ps, pe in p_blocks:
            ov_gt = [(gs, ge) for gs, ge in gt_blocks if overlaps(ps, pe, gs, ge)]
            if len(ov_gt) > 1: merge_c[act] += len(ov_gt) - 1

    insertion_c = Counter()
    deletion_c  = Counter()
    for act, p_blocks in pred_events.items():
        gt_blocks = gt_events.get(act, [])
        for ps, pe in p_blocks:
            if not any(overlaps(ps, pe, gs, ge) for gs, ge in gt_blocks):
                insertion_c[act] += pe - ps + 1
    for act, gt_blocks in gt_events.items():
        p_blocks = pred_events.get(act, [])
        for gs, ge in gt_blocks:
            if not any(overlaps(gs, ge, ps, pe) for ps, pe in p_blocks):
                deletion_c[act] += ge - gs + 1

    return {
        "insertion":      sum(insertion_c.values()),
        "deletion":       sum(deletion_c.values()),
        "over_underfill": sum(underfill_c.values()),
        "fragment_merge": sum(fragmentation_c.values()) + sum(merge_c.values()),
        "true_positives": sum(tp_counter.values()),
        "total_gt_minutes": total_gt_minutes,
    }

def collect_multilabel_metrics(a_type="gpt4-gloss", ground_truth="annotated_locomotion"):
    n = len(get_all_activities())
    y_true_all = np.empty((0, n)); y_pred_all = np.empty((0, n))
    for subject in participants:
        try:
            result = run_multilabel_metrics(subject, a_type=a_type, ground_truth=ground_truth)
            if result is None: continue
            y_true, y_pred = result
            y_true_all = np.vstack([y_true_all, y_true])
            y_pred_all = np.vstack([y_pred_all, y_pred])
        except Exception:
            continue
    if y_true_all.shape[0] == 0:
        return None
    return {
        "exact_match": exact_match_accuracy(y_true_all, y_pred_all),
        "hamming":     hamming_loss(y_true_all, y_pred_all),
        "f1":          f1_score(y_true_all, y_pred_all),
        "jaccard":     jaccard_index(y_true_all, y_pred_all),
    }

def plot_ablation():
    ground_truth = "annotated_locomotion"

    # --- collect data ---
    bar_conditions = {
        "no changes":       ("gpt4-gloss-info",      "oss-gloss-info"),
        "information loss": ("gpt4-gloss-temporal-exact",          "oss-gloss-temporal-exact"),
        "swapping order":  ("gpt4-gloss-swapping",  "oss-gloss-swapping"),
    }
    temporal_steps = ["info", "temporal-20", "temporal-40", "temporal-60"]
    temporal_conditions = {
        step: (f"gpt4-gloss-{step}", f"oss-gloss-{step}")
        for step in temporal_steps
    }

    print("Collecting bar chart data...")
    bar_data = {}
    for label, (gpt4_cond, oss_cond) in bar_conditions.items():
        bar_data[label] = {
            "oss": collect_multilabel_metrics(gpt4_cond, ground_truth),
            "gpt4":  collect_multilabel_metrics(oss_cond,  ground_truth),
        }

    print("Collecting temporal data...")
    temporal_data = {"gpt4": [], "oss": []}
    for step in temporal_steps:
        gpt4_cond, oss_cond = temporal_conditions[step]
        temporal_data["oss"].append(collect_multilabel_metrics(gpt4_cond, ground_truth))
        temporal_data["gpt4"].append(collect_multilabel_metrics(oss_cond,  ground_truth))

    # --- plot ---
    metric = "exact_match"
    metric_label = "Exact Match"

    GPT4_COLOR = "#305BAB"
    OSS_COLOR  = "#E07B39"
    bar_labels = list(bar_conditions.keys())
    x = np.arange(len(bar_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

    # --- subplot 1: grouped bar chart ---
    # 2 groups (gpt4, oss), each with 3 bars (gloss, info loss, swapping)
    n_bars = len(bar_labels)
    group_centers = np.array([0.0, 1.0 + n_bars * width])  # one center per model group
    palette = sns.light_palette("#305BAB", 3)

    colors = [palette[i] for i in range(n_bars)]
    condition_labels = bar_labels

    all_vals = []
    for gi, model in enumerate(["gpt4", "oss"]):
        offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * width
        for ci, lbl in enumerate(bar_labels):
            val = bar_data[lbl][model][metric] if bar_data[lbl][model] else 0
            all_vals.append(val)
            bar = ax1.bar(group_centers[gi] + offsets[ci], val, width,
                          color=colors[ci], edgecolor="black",
                          label=lbl if gi == 0 else "_nolegend_")
            ax1.text(group_centers[gi] + offsets[ci], val + 0.005,
                     f"{val:.2f}", ha='center', va='bottom', fontsize=13)

    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(["gpt-4o", "gpt-oss"], fontsize=12)
    ax1.set_ylabel(metric_label, fontsize=13)
    ax1.set_title("Impact of information loss\nand swapping order", fontsize=16)
    # set legend to have 3 columns and be centered above the plot
    ax1.legend(fontsize=13, loc='upper right', ncol=1, frameon=True)
    ax1.set_ylim(0.3, 1)

    # --- subplot 2: temporal degradation line chart ---
    x_labels = ["baseline (±0%)", "±20%", "±40%", "±60%"]
    gpt4_temp = [d[metric] if d else None for d in temporal_data["gpt4"]]
    oss_temp  = [d[metric] if d else None for d in temporal_data["oss"]]

    ax2.plot(x_labels, gpt4_temp, marker='o', color=GPT4_COLOR, label="gpt-4o", linewidth=2)
    ax2.plot(x_labels, oss_temp,  marker='o', color=OSS_COLOR,  label="gpt-oss",   linewidth=2)
    ax2.set_title("Impact of temporal jittering", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.set_ylim(0.3, max(v for v in gpt4_temp + oss_temp if v) * 1.2)
    for vals, color in [(gpt4_temp, GPT4_COLOR), (oss_temp, OSS_COLOR)]:
        for i, v in enumerate(vals):
            if v is not None:
                ax2.text(i, v + 0.005, f"{v:.2f}", ha='center', va='bottom', fontsize=11, color=color)
    #increase the size of x-axis labels
    ax2.set_xticklabels(x_labels, fontsize=13)
    plt.tight_layout()
    plt.savefig('/mnt/study/ari_work/llm-sensemaking/figs/ablation_plot.pdf', format='pdf', dpi=300)
    plt.show()
    print("Saved to figs/ablation_plot.pdf")

def plot_misalignment_analysis():
    data = {
        'Condition': ['oss-lst','gpt4-lst','oss-narrative','gpt4-narrative','oss-ps','gpt4-ps','oss-uEMA','gpt4-uEMA'],
        'Type':      ['gpt-oss','gpt-4o','gpt-oss','gpt-4o','gpt-oss','gpt-4o','gpt-oss','gpt-4o'],
        'Insertion':       [31.64,32.54,31.40,28.00,31.22,36.52,32.97,24.47],
        'Deletion':        [25.30,28.46,20.52,21.82,34.43,30.29,33.25,31.92],
        'Over/Underfill':  [6.29,6.06,6.07,4.27,6.40,4.96,4.67,7.22],
        'Fragment/Merge':  [1.62,1.32,1.60,1.55,1.19,1.27,1.02,1.60],
    }
    df = pd.DataFrame(data)
    df['ShortCondition'] = df['Condition'].str.replace('oss-|gpt4-', '', regex=True)
    df_melt = df.melt(id_vars=['Condition','Type','ShortCondition'], var_name='ErrorType', value_name='Percentage')
    short_order = ['ps','uEMA','lst','narrative']
    long_order  = ["passive sensing","μEMA","list-of-activities","narration-based"]
    palette = {k: mcolors.to_hex(c) for k, c in zip(short_order, sns.light_palette("#305BAB", len(short_order)))}

    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    for i, ctype in enumerate(['gpt-oss','gpt-4o']):
        for j, err in enumerate(['Insertion','Deletion','Over/Underfill','Fragment/Merge']):
            plt.subplot(2, 4, i*4 + j + 1)
            subset = df_melt[(df_melt['ErrorType']==err) & (df_melt['Type']==ctype)]
            sns.barplot(data=subset, x='ShortCondition', y='Percentage',
                        palette={k: palette[k] for k in subset['ShortCondition']},
                        order=short_order, edgecolor='black')
            plt.title(f'{ctype} - {err}', fontsize=20)
            plt.xlabel('')
            plt.ylabel('Percentage (%)' if j == 0 else '', fontsize=16)
            plt.ylim(0, subset['Percentage'].max() + 5)
            plt.yticks(fontsize=14)
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            plt.xticks([])

    handles = [mpatches.Patch(color=palette[k], label=long_order[short_order.index(k)]) for k in short_order]
    plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.5, 0), ncol=4, frameon=True, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('/mnt/study/ari_work/llm-sensemaking/figs/misalignment_analysis.pdf', format='pdf', dpi=300)

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

def calculate_fixed_amount(y_true, y_original, y_pred):
    y_true, y_original, y_pred = y_true.to_numpy(), y_original.to_numpy(), y_pred.to_numpy()
    same = better = worse = 0
    for i in range(y_true.shape[0]):
        if np.array_equal(y_pred[i], y_original[i]):
            same += 1
        else:
            pm = np.sum(y_pred[i] == y_true[i]); om = np.sum(y_original[i] == y_true[i])
            if pm > om:   better += 1
            elif pm < om: worse += 1
            else:         same += 1
    return same, better, worse, y_true.shape[0]

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

def run_multilabel_metrics_for_everyone(a_type="gpt4-gloss", ground_truth="thigh_sensor"):
    n = len(get_all_activities())
    y_true_all = np.empty((0, n)); y_pred_all = np.empty((0, n))
    for subject in participants:
        try:
            result = run_multilabel_metrics(subject, a_type=a_type, ground_truth=ground_truth)
            if result is None: continue
            y_true, y_pred = result
            y_true_all = np.vstack([y_true_all, y_true])
            y_pred_all = np.vstack([y_pred_all, y_pred])
        except Exception:
            continue
    if y_true_all.shape[0] == 0:
        print("No valid data found."); return
    print(f"CONDITION: {a_type}: Exact Match: {exact_match_accuracy(y_true_all, y_pred_all):.4f}, "
          f"Hamming: {hamming_loss(y_true_all, y_pred_all):.4f}, "
          f"F1: {f1_score(y_true_all, y_pred_all):.4f}, "
          f"Jaccard: {jaccard_index(y_true_all, y_pred_all):.4f}")

def run_fixed_amount_for_everyone(a_type="gpt4-gloss"):
    unchanged_l = better_l = worse_l = []
    for subject in participants:
        try:
            fixed    = pd.read_csv(os.path.join(BASE, f'fixed/{subject}_{a_type}_merged.csv'), index_col=0)
            original = pd.read_csv(os.path.join(BASE, f'fixed/{subject}_original_merged.csv'), index_col=0)
            original = original.rename(columns={'activity': 'activity_original', 'locomotion': 'locomotion_original'})
            merged   = pd.merge(fixed, original, on='timestamp', how='right').fillna('')
            y_true, y_orig, y_pred = process_multilabel([merged[['locomotion_original']], merged[['activity_original']], merged[['activity']]])
            u, b, w, _ = calculate_fixed_amount(y_true, y_orig, y_pred)
            unchanged_l.append(u); better_l.append(b); worse_l.append(w)
        except Exception:
            continue
    print(f"CONDITION: {a_type}: Unchanged: {np.mean(unchanged_l):.2f}, Better: {np.mean(better_l):.2f}, Worse: {np.mean(worse_l):.2f}")


if __name__ == "__main__":
    lst_conditions = ["gpt4-gloss-info","oss-gloss-info","gpt4-gloss-swapping","oss-gloss-swapping",
                      "gpt4-gloss-temporal-20", "gpt4-gloss-temporal-exact","oss-gloss-temporal-exact", 
                      "oss-gloss-temporal-20","gpt4-gloss-temporal-40","oss-gloss-temporal-40",
                      "gpt4-gloss-temporal-60","oss-gloss-temporal-60"]
    # lst_conditions = ["gpt4-lst-hard", "oss-lst-hard", "gpt4-lst-hard-no-memory", "oss-lst-hard-no-memory",
    #                 "gpt4-lst-hard-no-presentation", "oss-lst-hard-no-presentation",
    #                 "gpt4-lst", "oss-lst", "gpt4-lst-no-memory", "oss-lst-no-memory",
    #                 "gpt4-lst-no-presentation", "oss-lst-no-presentation",
    #                 "gpt4-ps", "oss-ps", "gpt4-ps-no-memory", "oss-ps-no-memory",
    #                 "gpt4-ps-no-presentation", "oss-ps-no-presentation",
    #                 "gpt4-gloss", "oss-gloss", "gpt4-gloss-no-memory", "oss-gloss-no-memory",
    #                 "gpt4-gloss-no-presentation", "oss-gloss-no-presentation",
    #                 "gpt4-uEMA", "oss-uEMA", "gpt4-uEMA-no-memory", "oss-uEMA-no-memory",
    #                 "gpt4-uEMA-no-presentation", "oss-uEMA-no-presentation",]

    for condition in lst_conditions:
        run_multilabel_metrics_for_everyone(a_type=condition, ground_truth="annotated_locomotion")
    # plot_ablation()