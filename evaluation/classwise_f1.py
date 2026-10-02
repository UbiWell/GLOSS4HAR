import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sklearn.metrics
import os

# --- reuse from correct_labels_check.py ---
from correct_labels_check import (
    participants, get_all_results_from_participants, parse_results_column,
    read_annotated_file, clean_activity_labels, split_into_minute_level,
    get_all_activities, process_multilabel, BASE
)


def get_classwise_f1(a_type, ground_truth="annotated_locomotion"):
    """Returns per-class F1 scores as a Series indexed by activity label."""
    all_activities = sorted(get_all_activities())
    y_true_all = np.empty((0, len(all_activities)))
    y_pred_all = np.empty((0, len(all_activities)))

    for subject in participants:
        try:
            result_df = get_all_results_from_participants(subject, a_type=a_type)
            if result_df.empty:
                continue
            result_df = parse_results_column(result_df, 'result')
            result_df = clean_activity_labels(result_df, 'activity')
            result_df = split_into_minute_level(result_df)

            if ground_truth == "annotated_locomotion":
                gt_df = clean_activity_labels(read_annotated_file(subject), 'activity')
                gt_df = split_into_minute_level(gt_df).rename(columns={'activity': 'locomotion'})
            gt_df = gt_df[['timestamp', 'locomotion']]

            result_df = result_df.groupby('timestamp').agg({'activity': lambda x: ', '.join(x)}).reset_index()
            result_df['activity'] = result_df['activity'].apply(
                lambda x: ', '.join(sorted(set(i.strip() for i in x.split(',')))))

            merged = pd.merge(gt_df, result_df, on='timestamp', how='inner')
            merged = merged[(merged['locomotion'] != '') & (merged['activity'] != '')]

            y_true, y_pred = process_multilabel([merged[['locomotion']], merged[['activity']]])
            y_true_all = np.vstack([y_true_all, y_true])
            y_pred_all = np.vstack([y_pred_all, y_pred])
        except Exception as e:
            print(f"  Skipping {subject} for {a_type}: {e}")
            continue

    if y_true_all.shape[0] == 0:
        return None

    f1_scores = sklearn.metrics.f1_score(y_true_all, y_pred_all, average=None, zero_division=0)
    return pd.Series(f1_scores, index=all_activities)


def filter_active_classes(f1_dict, min_support=0.01):
    """Keep only classes that have nonzero F1 in at least one condition."""
    df = pd.DataFrame(f1_dict)
    active = df[(df > min_support).any(axis=1)]
    return active.sort_values(by=df.columns[0], ascending=False)


def plot_classwise_f1(ax, data_df, conditions, colors, title, labels=None):
    classes = data_df.index.tolist()
    n_classes = len(classes)
    n_conds = len(conditions)
    x = np.arange(n_classes)
    width = 0.8 / n_conds
    offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width

    for i, cond in enumerate(conditions):
        vals = data_df[cond].values if cond in data_df.columns else np.zeros(n_classes)
        lbl = labels[cond] if labels and cond in labels else cond
        ax.bar(x + offsets[i], vals, width, label=lbl,
               color=colors[cond], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=16)
    ax.set_ylabel("F1 Score", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)


# ── Color scheme ──────────────────────────────────────────────────────────────

p1_colors = {
    "gpt4-vanilla": "#AEC6E8", "gpt4-gloss": "#305BAB",
    "oss-vanilla":  "#AEC6E8", "oss-gloss":  "#305BAB",
}
p1_labels = {
    "gpt4-vanilla": "RAG", "gpt4-gloss": "GLOSS",
    "oss-vanilla":  "RAG", "oss-gloss":  "GLOSS",
}

# ── Plot 1: vanilla vs gloss ──────────────────────────────────────────────────

p1_conds = ["gpt4-vanilla", "gpt4-gloss", "oss-vanilla", "oss-gloss"]

print("Computing Plot 1 (vanilla vs gloss)...")
f1_p1 = {}
for cond in p1_conds:
    print(f"  {cond}...")
    f1_p1[cond] = get_classwise_f1(cond)
f1_p1 = {k: v for k, v in f1_p1.items() if v is not None}

df_p1_gpt4 = filter_active_classes({k: v for k, v in f1_p1.items() if k.startswith("gpt4")})
df_p1_oss  = filter_active_classes({k: v for k, v in f1_p1.items() if k.startswith("oss")})
shared_classes_p1 = df_p1_gpt4.index.union(df_p1_oss.index)
df_p1_gpt4 = df_p1_gpt4.reindex(shared_classes_p1, fill_value=0)
df_p1_oss  = df_p1_oss.reindex(shared_classes_p1, fill_value=0)

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
plot_classwise_f1(ax1a, df_p1_gpt4, ["gpt4-vanilla", "gpt4-gloss"], p1_colors, "gpt-oss",  labels=p1_labels)
plot_classwise_f1(ax1b, df_p1_oss,  ["oss-vanilla",  "oss-gloss"],  p1_colors, "gpt-4o", labels=p1_labels)

legend_handles_p1 = [
    mpatches.Patch(color="#AEC6E8", label="RAG"),
    mpatches.Patch(color="#305BAB", label="GLOSS"),
]
fig1.legend(handles=legend_handles_p1, loc='lower center', ncol=2, fontsize=16,
            frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(os.path.join('/mnt/study/ari_work/llm-sensemaking/figs', 'classwise_f1_vanilla_vs_gloss.pdf'),
            format='pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved classwise_f1_vanilla_vs_gloss.pdf")


# ── Plot 2: ps / uEMA / lst / lst-hard ───────────────────────────────────────

task_keys = ["ps", "uEMA", "lst", "lst-hard"]
task_labels = {
    "ps":       "no self-report",
    "uEMA":     "uEMA",
    "lst":      "list-of-activity (approx time)",
    "lst-hard": "list-of-activity (no time)",
}
BLUES     = sns.light_palette("#305BAB", len(task_keys) + 1)[1:]
p2_colors = {f"gpt4-{k}": BLUES[i] for i, k in enumerate(task_keys)}
p2_colors.update({f"oss-{k}": BLUES[i] for i, k in enumerate(task_keys)})
p2_labels_gpt4 = {f"gpt4-{k}": task_labels[k] for k in task_keys}
p2_labels_oss  = {f"oss-{k}":  task_labels[k] for k in task_keys}

p2_conds = [f"gpt4-{k}" for k in task_keys] + [f"oss-{k}" for k in task_keys]

print("Computing Plot 2 (ps / uEMA / lst / lst-hard)...")
f1_p2 = {}
for cond in p2_conds:
    print(f"  {cond}...")
    f1_p2[cond] = get_classwise_f1(cond)
f1_p2 = {k: v for k, v in f1_p2.items() if v is not None}

df_p2_gpt4 = filter_active_classes({k: v for k, v in f1_p2.items() if k.startswith("gpt4")})
df_p2_oss  = filter_active_classes({k: v for k, v in f1_p2.items() if k.startswith("oss")})
shared_classes_p2 = df_p2_gpt4.index.union(df_p2_oss.index)
df_p2_gpt4 = df_p2_gpt4.reindex(shared_classes_p2, fill_value=0)
df_p2_oss  = df_p2_oss.reindex(shared_classes_p2, fill_value=0)

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
plot_classwise_f1(ax2a, df_p2_gpt4, [f"gpt4-{k}" for k in task_keys], p2_colors, "gpt-oss",  labels=p2_labels_gpt4)
plot_classwise_f1(ax2b, df_p2_oss,  [f"oss-{k}"  for k in task_keys], p2_colors, "gpt-40", labels=p2_labels_oss)

legend_handles_p2 = [mpatches.Patch(color=BLUES[i], label=task_labels[k]) for i, k in enumerate(task_keys)]
fig2.legend(handles=legend_handles_p2, loc='lower center', ncol=len(task_keys), fontsize=16,
            frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(os.path.join('/mnt/study/ari_work/llm-sensemaking/figs', 'classwise_f1_conditions.pdf'),
            format='pdf', dpi=300, bbox_inches='tight')
plt.show()
print("Saved classwise_f1_conditions.pdf")