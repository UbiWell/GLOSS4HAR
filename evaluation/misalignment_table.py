import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from scipy import stats
from correct_labels_check import (
    participants, get_all_results_from_participants, parse_results_column,
    read_annotated_file, mapping_timeline, clean_activity_labels,
    get_all_activities, process_multilabel
)

CONDITIONS = {
    "gpt4-ps":       "gpt-4o | no self-report",
    "oss-ps":        "gpt-oss | no self-report",
    "gpt4-uEMA":     "gpt-4o | uEMA",
    "oss-uEMA":      "gpt-oss | uEMA",
    "gpt4-lst":      "gpt-4o | list (approx time)",
    "oss-lst":       "gpt-oss | list (approx time)",
    "gpt4-lst-hard": "gpt-4o | list (no time)",
    "oss-lst-hard":  "gpt-oss | list (no time)",
}


def ward_segment_scoring(gt_seq, pred_seq):
    n = len(gt_seq)
    segments = []
    i = 0
    while i < n:
        gt_v, pred_v = gt_seq[i], pred_seq[i]
        j = i + 1
        while j < n and gt_seq[j] == gt_v and pred_seq[j] == pred_v:
            j += 1
        segments.append({'gt': gt_v, 'pred': pred_v, 'start': i, 'end': j - 1})
        i = j

    def basic_score(seg):
        g, p = seg['gt'], seg['pred']
        if g == 1 and p == 1: return 'TP'
        if g == 0 and p == 0: return 'TN'
        if g == 0 and p == 1: return 'FP'
        return 'FN'

    for seg in segments:
        seg['score'] = basic_score(seg)

    counts = defaultdict(int)
    m = len(segments)
    for idx, seg in enumerate(segments):
        score = seg['score']
        prev_score = segments[idx - 1]['score'] if idx > 0     else None
        next_score = segments[idx + 1]['score'] if idx < m - 1 else None
        length = seg['end'] - seg['start'] + 1

        if score == 'TP':
            counts['TP'] += length
        elif score == 'TN':
            counts['TN'] += length
        elif score == 'FP':
            prev_tp = prev_score == 'TP'
            next_tp = next_score == 'TP'
            if prev_tp and next_tp:       counts['Merge']    += length
            elif prev_tp or next_tp:      counts['Overfill'] += length
            else:                         counts['Insertion'] += length
        elif score == 'FN':
            prev_tp = prev_score == 'TP'
            next_tp = next_score == 'TP'
            if prev_tp and next_tp:       counts['Fragmentation'] += length
            elif prev_tp or next_tp:      counts['Underfill']     += length
            else:                         counts['Deletion']       += length

    return counts


def score_condition(a_type):
    all_activities = sorted(get_all_activities())
    totals = defaultdict(int)
    per_subject = []
    per_activity = []

    for subject in participants:
        df = get_all_results_from_participants(subject, a_type=a_type)
        if df.empty:
            continue
        df = parse_results_column(df, 'result')
        df = clean_activity_labels(df, 'activity')
        ann = read_annotated_file(subject)
        ann = clean_activity_labels(ann, 'activity')
        mapped = mapping_timeline(df, ann)

        y_gt, y_pred = process_multilabel(
            [mapped[['annotated_activity']], mapped[['result_activity']]]
        )
        y_gt   = y_gt.to_numpy()
        y_pred = y_pred.to_numpy()

        active_cols = np.where((y_gt.sum(axis=0) > 0) | (y_pred.sum(axis=0) > 0))[0]
        subj_counts = defaultdict(int)

        for a_idx in active_cols:
            gt_seq   = y_gt[:, a_idx].tolist()
            pred_seq = y_pred[:, a_idx].tolist()
            counts = ward_segment_scoring(gt_seq, pred_seq)
            for k, v in counts.items():
                totals[k] += v
                subj_counts[k] += v
            act_total = sum(counts.values())
            if act_total > 0:
                per_activity.append(dict(counts))

        per_subject.append(dict(subj_counts))

    return totals, per_subject, per_activity


# ── Main table ────────────────────────────────────────────────────────────────

rows = []
all_per_activity = {}

for a_type, label in CONDITIONS.items():
    print(f"Scoring {a_type}...")
    totals, per_subject, per_activity = score_condition(a_type)
    all_per_activity[a_type] = per_activity

    total = sum(totals.values())
    if total == 0:
        continue

    tp_tn         = totals['TP'] + totals['TN']
    under_overfill = totals['Underfill'] + totals['Overfill']
    frag_merge     = totals['Fragmentation'] + totals['Merge']

    rows.append({
        "Condition":      label,
        "TP+TN":          f"{tp_tn/total*100:.1f}%",
        "Insertion":      f"{totals['Insertion']/total*100:.1f}%",
        "Deletion":       f"{totals['Deletion']/total*100:.1f}%",
        "Under/Overfill": f"{under_overfill/total*100:.1f}%",
        "Frag/Merge":     f"{frag_merge/total*100:.1f}%",
        "Sum":            f"{total/total*100:.1f}%",
    })

df_results = pd.DataFrame(rows).set_index("Condition")
df_gpt4 = df_results[df_results.index.str.startswith("gpt-4o")].copy()
df_oss  = df_results[df_results.index.str.startswith("gpt-oss")].copy()
df_gpt4.index = df_gpt4.index.str.replace("gpt-4o | ", "", regex=False)
df_oss.index  = df_oss.index.str.replace("gpt-oss | ", "", regex=False)

print("\n" + "=" * 80)
print("Misalignment Analysis (Ward et al. 2006 segment scoring)")
print("TP + TN + Insertion + Deletion + Under/Overfill + Frag/Merge = 100%")
print("=" * 80)
print("\n--- gpt-4o ---")
print(df_gpt4.to_string())
print("\n--- gpt-oss ---")
print(df_oss.to_string())


# ── Statistical tests ─────────────────────────────────────────────────────────

def get_rates(per_activity_list, error_key):
    """Rate = error_type / total_errors (excluding TP/TN) per activity instance."""
    rates = []
    for act in per_activity_list:
        total_errors = (act.get('Insertion', 0) + act.get('Deletion', 0) +
                        act.get('Underfill', 0) + act.get('Overfill', 0) +
                        act.get('Fragmentation', 0) + act.get('Merge', 0))
        if total_errors == 0:
            continue
        if error_key == 'Under/Overfill':
            val = act.get('Underfill', 0) + act.get('Overfill', 0)
        elif error_key == 'Frag/Merge':
            val = act.get('Fragmentation', 0) + act.get('Merge', 0)
        else:
            val = act.get(error_key, 0)
        rates.append(val / total_errors)
    return rates


print("\n" + "=" * 80)
print("Statistical Tests: list vs non-list conditions")
print("(Mann-Whitney U, per-activity error proportions out of total errors)")
print("=" * 80)

for model_prefix, model_name in [("gpt4", "gpt-4o"), ("oss", "gpt-oss")]:
    nonlst_types      = [f"{model_prefix}-ps", f"{model_prefix}-uEMA"]
    nonlst_activities = [a for t in nonlst_types for a in all_per_activity[t]]

    for lst_type, lst_label in [(f"{model_prefix}-lst",      "list (approx time)"),
                                 (f"{model_prefix}-lst-hard", "list (no time)")]:
        lst_activities = all_per_activity[lst_type]

        print(f"\n--- {model_name} | {lst_label} vs non-list ---")
        print(f"{'Error':<20} {'List (mean)':>14} {'Non-list (mean)':>16} {'U stat':>10} {'p-value':>10} {'sig':>5}")
        print("-" * 80)

        for err in ['Insertion', 'Deletion', 'Under/Overfill', 'Frag/Merge']:
            lst_rates    = get_rates(lst_activities, err)
            nonlst_rates = get_rates(nonlst_activities, err)
            u_stat, p_val = stats.mannwhitneyu(lst_rates, nonlst_rates, alternative='two-sided')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f"{err:<20} {np.mean(lst_rates):>14.4f} {np.mean(nonlst_rates):>16.4f} "
                  f"{u_stat:>10.1f} {p_val:>10.4f} {sig:>5}")