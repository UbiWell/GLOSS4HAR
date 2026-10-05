"""Run GLOSS4HAR on the two tasks in the paper.

Task 1 - correct participant annotations:
    python GLOSS4HAR.py correct --annotations annotations/pilot2_cleaned.csv

Task 2 - build an activity timeline from passive sensing and a low-effort self-report:
    python GLOSS4HAR.py timeline --subject pilot2 --self-report none
    python GLOSS4HAR.py timeline --subject pilot2 --self-report uema
    python GLOSS4HAR.py timeline --subject pilot2 --self-report list --activity-list lists/pilot2.csv
    python GLOSS4HAR.py timeline --subject pilot2 --self-report list-no-time --activity-list lists/pilot2.csv

Common options: --model {gpt-4o,gpt-5,gpt-oss}, --no-memory, --no-presentation, --output.
Runs are resumable: rows/hours already in the output file are skipped.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agents')))

participants = {
    'pilot2': '2025/02/19',
    'pilot8': '2025/02/27',
    'pilot5': '2025/02/25',
    'pilot6': '2025/02/25',
    'pilot7': '2025/02/27',
    'pilot9': '2025/03/02',
    'pilot10': '2025/03/10',
    'pilot11': '2025/03/03',
}

one_hours = ['8am-9am', '9am-10am', '10am-11am', '11am-12pm', '12pm-1pm', '1pm-2pm', '2pm-3pm', '3pm-4pm', '4pm-5pm', '5pm-6pm', '6pm-7pm', '7pm-8pm', '8pm-9pm', '9pm-10pm', '10pm-11pm']

SELF_REPORTS = ['none', 'uema', 'list', 'list-no-time']

RESULT_COLUMNS = ['subject', 'date', 'uncertainty_start', 'uncertainty_end', 'labels', 'result', 'action_plan',
                  'memory', 'understanding', 'information_requests', 'function_calls']

MEMORY_HEADER = "This is the previous annotations (have been correct). Do not include these in final answer:\n"


def get_presentation_format():
    from agents.constants import ABLATION_PRESENTATION_AGENT

    if ABLATION_PRESENTATION_AGENT:
        return "- include both posture and activity in your responses"
    return """
                            - Only use the postures among this list: 'sitting', 'standing', 'lying down', 'reclining', 'upright'
                            - Only use the activities among this list: 'video gaming',  'walking', 'stair climbing',  'getting ready',  'driving',  'bicycling',  'vigorous bicycling',  'aerobics',  'cleaning',  'cooking',  'laundry',  'playing with pet',  'listening to music',  'watching movies/TV',  'studying',  'reading',  'riding in car',  'riding train',  'riding bus',  'playing musical instruments',  'attending meeting',  'computer using', 'phone using',  'running',  'getting dressed',  'grooming',  'using bathroom',  'eating',  'talking','strength training',  'washing dishes',  'carrying groceries',  'putting away groceries',  'shopping',  'making bed',  'packing/unpacking',  'sleeping',  'playing sports'
                            """


def run_sensemaker(query, memory=''):
    import sensemaking_process
    from agents.constants import ABLATION_MEMORY

    sensemaker = sensemaking_process.SenseMaker(query, get_presentation_format())
    if memory != '' and not ABLATION_MEMORY:
        sensemaker.memory = MEMORY_HEADER + memory
    sensemaker.make_sense()
    return sensemaker


def run_correction(annotations_csv, output_csv):
    """Task 1: correct each annotation row (columns: subject, date, labels, uncertainty_start, uncertainty_end)."""
    from prompts import get_correction_prompt

    df = pd.read_csv(annotations_csv)

    if os.path.exists(output_csv):
        done_df = pd.read_csv(output_csv)
        for col in ('result', 'thinking'):
            if col in done_df.columns:
                df[col] = done_df[col]
    else:
        df['result'] = None
        df['thinking'] = None
        df.to_csv(output_csv, index=False)

    memory_past_annotations = ''

    for index, row in df.iterrows():
        if pd.notna(df.at[index, 'result']):
            memory_past_annotations += f"{df.at[index, 'result']}\n"
            continue

        query = get_correction_prompt(
            subject_id=row['subject'],
            activity=row['labels'],
            time_period=f"{row['uncertainty_start']}-{row['uncertainty_end']}",
            participants=participants,
        )
        result = run_sensemaker(query, memory=memory_past_annotations)

        try:
            df.at[index, 'result'] = str(result.answer)
            df.at[index, 'thinking'] = str(result.understanding)
            memory_past_annotations += f"{df.at[index, 'result']}\n"
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            df.at[index, 'result'] = 'Error processing row'
            df.at[index, 'thinking'] = ''

        df.to_csv(output_csv, index=False)

    return df


def build_timeline_query(subject, time_period, self_report, activities):
    from prompts import get_triangulation_prompt, get_list_triangulation_prompt

    if self_report in ('none', 'uema'):
        # for uEMA the self-reports come from the uEMA database, not the prompt
        return get_triangulation_prompt(subject_id=subject, time_period=time_period, participants=participants)
    return get_list_triangulation_prompt(subject_id=subject, time_period=time_period, participants=participants,
                                         activities=activities, with_time=(self_report == 'list'))


def run_timeline(subject, self_report, output_csv, activity_list=None):
    """Task 2: generate the activity timeline for `subject`, one hour at a time."""
    activities = ''
    if self_report in ('list', 'list-no-time'):
        with open(activity_list, 'r') as f:
            activities = f.read()

    if not os.path.exists(output_csv):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(output_csv, index=False)
    results_df = pd.read_csv(output_csv)

    memory_past_annotations = ''

    for time_period in one_hours:
        start, end = time_period.split('-')
        if ((results_df['subject'] == subject) & (results_df['uncertainty_start'] == start) & (results_df['uncertainty_end'] == end)).any():
            continue

        try:
            result = run_sensemaker(build_timeline_query(subject, time_period, self_report, activities),
                                    memory=memory_past_annotations)
            new_row = {
                'subject': subject,
                'date': participants[subject],
                'uncertainty_start': start,
                'uncertainty_end': end,
                'labels': '',  # no labels for this task
                'result': str(result.answer),
                'action_plan': str(result.hypothesis),
                'memory': str(result.memory),
                'understanding': str(result.understanding),
                'information_requests': str(result.information_request),
                'function_calls': str(result.function_calls)
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            memory_past_annotations += f"{new_row['result']}\n"
        except Exception as e:
            print(f"Error processing time period {time_period}: {e}")
        results_df.to_csv(output_csv, index=False)

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--model', choices=['gpt-4o', 'gpt-5', 'gpt-oss'], default='gpt-oss')
    common.add_argument('--no-memory', action='store_true', help='ablation: do not pass earlier results as memory')
    common.add_argument('--no-presentation', action='store_true', help='ablation: disable the presentation agent')
    common.add_argument('--output', help='results CSV (default: results/<task>_..._<model>.csv)')

    sub = parser.add_subparsers(dest='task', required=True)
    correct = sub.add_parser('correct', parents=[common], help='task 1: correct participant annotations')
    correct.add_argument('--annotations', required=True, help='CSV of annotations to correct')

    timeline = sub.add_parser('timeline', parents=[common], help='task 2: timeline from passive sensing + self-reports')
    timeline.add_argument('--subject', required=True, choices=sorted(participants))
    timeline.add_argument('--self-report', required=True, choices=SELF_REPORTS)
    timeline.add_argument('--activity-list', help="participant's activity list (required for list / list-no-time)")

    args = parser.parse_args()
    if args.task == 'timeline' and args.self_report.startswith('list') and not args.activity_list:
        parser.error('--activity-list is required for --self-report list / list-no-time')
    return args


def main():
    args = parse_args()

    # The agents read their configuration from these at import time, so set them before importing anything.
    os.environ['GLOSS4HAR_MODEL'] = args.model
    os.environ['GLOSS4HAR_NO_MEMORY'] = str(args.no_memory)
    os.environ['GLOSS4HAR_NO_PRESENTATION'] = str(args.no_presentation)
    os.environ['GLOSS4HAR_USE_UEMA'] = str(args.task == 'timeline' and args.self_report == 'uema')

    suffix = args.model + ('_no_memory' if args.no_memory else '') + ('_no_presentation' if args.no_presentation else '')
    if args.task == 'correct':
        name = os.path.splitext(os.path.basename(args.annotations))[0]
        output = args.output or os.path.join('results', f'correct_{name}_{suffix}.csv')
    else:
        output = args.output or os.path.join('results', f'timeline_{args.self_report}_{args.subject}_{suffix}.csv')
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    if args.task == 'correct':
        run_correction(args.annotations, output)
    else:
        run_timeline(args.subject, args.self_report, output, args.activity_list)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
