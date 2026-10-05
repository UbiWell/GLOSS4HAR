# GLOSS4HAR

Code and data for **"Feasibility of Using a Multi-Agent LLM System to Correct Annotations and Support Low-Effort
Activity Labeling"** (Le, Choube, Mishra, Intille — *Proc. ACM IMWUT* 10(3), 2026,
[doi:10.1145/3831653](https://doi.org/10.1145/3831653)).

GLOSS4HAR is a multi-agent LLM system that combines passive smartphone/smartwatch sensing data with participants'
self-reports to help researchers clean activity annotations. The paper evaluates it on two tasks:

1. **Annotation correction:** correct and reconcile participants' own activity annotations.
2. **Timeline generation from low-effort self-reports:** triangulate passive sensing with lightweight self-reports
   (none, uEMA, or an activity list with or without times) to produce a minute-level posture/activity timeline.

## How it works

`sensemaking_process.SenseMaker` runs one query through the agents in `agents/`:
action plan → information seeking → database manager → coding agent → sensemaking → presentation.
The coding agent writes Python that calls the data-stream functions in `data_streams/` and runs it in a Docker
container. Those functions read the CSVs in `data/`. All agents use OpenAI gpt-4o.

## Setup

Requires Python 3.11+, Docker, and an OpenAI API key.

```bash
pip install -r requirements.txt
docker build -t sensemaking-code .     # image the coding agent runs generated code in
export OPENAI_API_KEY=...
```

Optional: `GOOGLE_API_KEY` for reverse geocoding in the location functions.

## Data

`data/` holds one CSV per sensor stream for the 8 participants in the paper (`pilot2`, `pilot5`–`pilot11`).
Every file has `subject_id` and `timestamp` (epoch milliseconds) columns plus its value column(s):

| File | Value column | Source | Agent database |
|---|---|---|---|
| `android_phone_usage.csv` | `in_use` | phone screen in use | phone usage |
| `garmin_hr.csv` | `heart_rate` | watch heart rate | heart rate |
| `pixel_ambient_noise.csv` | `ambient_noise` | watch ambient sound classes | ambient noise |
| `pixel_skin_temperature.csv` | `skin_temp` | watch skin temperature | skin temperature |
| `pixel_steps.csv` | `steps` | watch step count | step count |
| `pixel_wear_detection.csv` | `wear_detection` | watch on-wrist detection | watch wear |
| `pixel_wrist_auc.csv` | `total_auc` | watch accelerometer activity counts | (not used by the agents in the paper) |
| `uEMA.csv` | `uEMA` | watch micro-EMA self-reports (some voice-transcribed) | uEMA (only with `--self-report uema`) |

**Location data is not included.** The paper's runs also used phone GPS (`android_location.csv`: `latitude`,
`longitude`), but the raw traces can identify participants. Without that file the location database is turned off
(a warning is printed), so results will differ from the paper. If you have access to it, put it in `data/`.

The participants' annotation files (task 1 input) and activity lists (task 2 input) are not included either.

The data folder can be moved with `GLOSS4HAR_DATA_DIR` when running without Docker. The container only sees the
repository, so keep the data in `data/` for full runs.

## Running

Outputs go to `results/` unless you pass `--output`. Runs resume: rows/hours already in the output file are skipped.

**Task 1: annotation correction.** The input CSV has columns `subject, date, labels, uncertainty_start, uncertainty_end`.

```bash
python GLOSS4HAR.py correct --annotations annotations/pilot2_cleaned.csv
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

**Baseline:** the single-LLM ("vanilla") annotation-correction baseline puts summarized sensor data straight into one
gpt-4o prompt:

```bash
python RAG4HAR.py --annotations annotations/pilot2_cleaned.csv
```

## License

Code is released under the MIT License (see `LICENSE`).

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
