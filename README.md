# GLOSS4HAR

Code for **"Feasibility of Using a Multi-Agent LLM System to Correct Annotations and Support Low-Effort Activity Labeling"**
(Le, Choube, Mishra, Intille — *Proc. ACM IMWUT* 10(3), 2026, [doi:10.1145/3831653](https://doi.org/10.1145/3831653)).

GLOSS4HAR is a multi-agent LLM system that combines passive smartphone/smartwatch sensing data with participants'
self-reports to help researchers clean activity annotations. The paper evaluates it on two tasks:

1. **Annotation correction:** correct and reconcile participants' own activity annotations.
2. **Timeline generation from low-effort self-reports:** triangulate passive sensing with lightweight self-reports
   (none, uEMA, or an activity list with or without times) to produce a minute-level posture/activity timeline.

## How it works

`sensemaking_process.SenseMaker` runs one query through the agents in `agents/`:
action plan → information seeking → database manager → coding agent → sensemaking → presentation.
The coding agent writes Python that calls the data-stream functions in `data_streams/` and runs it in a Docker
container. Those functions read the CSVs in `data/` through `data_processing/data_processing_utils.py`.

## Setup

Requires Python 3.11+ and Docker.

```bash
pip install -r requirements.txt            # pipeline
pip install -r requirements-analysis.txt   # + evaluation/plotting scripts
docker build -t sensemaking-code .         # image the coding agent runs generated code in
```

Pick an LLM backend:

| `--model` | Needs |
|---|---|
| `gpt-4o`, `gpt-5` | `OPENAI_API_KEY` |
| `gpt-oss` (default) | an Ollama server with `gpt-oss:20b`; set `LLM_API_URL` (default `http://localhost:11434/api/generate`) and optionally `LLM_MODEL` |

Optional: `GOOGLE_API_KEY` for reverse geocoding in the location tools.

## Data

Put one CSV per data stream in `data/` (or point `GLOSS4HAR_DATA_DIR` at another folder when not using Docker; the
container only sees the repository, so keep the data in `data/` for full runs). Every file has `subject_id` and
`timestamp` (epoch milliseconds) columns plus the stream's value column:

| File | Value column | Source |
|---|---|---|
| `android_location.csv` | `latitude`, `longitude` | phone GPS |
| `android_phone_usage.csv` | `in_use` | phone screen in use |
| `garmin_hr.csv` | `heart_rate` | watch heart rate |
| `pixel_ambient_noise.csv` | `ambient_noise` | watch ambient sound classes |
| `pixel_skin_temperature.csv` | `skin_temp` | watch skin temperature |
| `pixel_steps.csv` | `steps` | watch step count |
| `pixel_wear_detection.csv` | `wear_detection` | watch on-wrist detection |
| `pixel_wrist_auc.csv` | `total_auc` | watch accelerometer activity counts |
| `uEMA.csv` | `uEMA` | watch micro-EMA self-reports (uEMA condition only) |

The study data are not included in this repository.

## Running

Outputs go to `results/` unless you pass `--output`. Runs resume: rows/hours already in the output file are skipped.

**Task 1: annotation correction.** The input CSV has columns `subject, date, labels, uncertainty_start, uncertainty_end`.

```bash
python GLOSS4HAR.py correct --annotations annotations/pilot2_cleaned.csv --model gpt-4o
```

**Task 2: timeline generation.** Runs each hour from 8am to 11pm on the participant's study day.

| Condition | Command |
|---|---|
| Passive sensing only | `python GLOSS4HAR.py timeline --subject pilot2 --self-report none` |
| uEMA | `python GLOSS4HAR.py timeline --subject pilot2 --self-report uema` |
| Activity list, approximate times | `python GLOSS4HAR.py timeline --subject pilot2 --self-report list --activity-list lists/pilot2.csv` |
| Activity list, no times | `python GLOSS4HAR.py timeline --subject pilot2 --self-report list-no-time --activity-list lists/pilot2.csv` |

The activity-list file is inserted into the prompt as is.

**Ablations** (either task): `--no-memory` stops earlier results being passed as memory, and `--no-presentation`
swaps the presentation agent and the fixed posture/activity vocabulary for a generic answer prompt.

**Baseline:** `RAG4HAR.py` runs the single-LLM ("vanilla") annotation-correction baseline. Its input files are
still listed in its `__main__` block.

## Evaluation

`correct_labels_check.py`, `ttest_analysis.py`, `evaluation/classwise_f1.py` and `evaluation/misalignment_table.py`
compute the F1 scores, statistics and figures in `figs/` from the result CSVs. They still expect the original
results folder layout (`BASE` in `correct_labels_check.py`).

## Citation

```bibtex
@article{le2026gloss4har,
  title   = {Feasibility of Using a Multi-Agent LLM System to Correct Annotations and Support Low-Effort Activity Labeling},
  author  = {Le, Ha and Choube, Akshat and Mishra, Varun and Intille, Stephen S.},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume  = {10},
  number  = {3},
  pages   = {1--40},
  year    = {2026},
  doi     = {10.1145/3831653}
}
```
