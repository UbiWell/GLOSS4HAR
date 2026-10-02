import pandas as pd
import numpy as np
import os, sys
import traceback

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

total_annotations = 0
total_bunching_errors = 0
total_omission_errors = 0
total_temporal_errors = 0

for participant in participants:
    path = f"/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/error_flagging/{participant}_cleaned.xlsx"
    df = pd.read_excel(path)
    # map 'omit secondary activities' to 'missing secondary activities', and add a 'missing short activities' column initialized to False
    df = df.rename(columns={'omit secondary activities': 'missing secondary activities', "omit activities": 'missing short activities'})
    if 'missing short activities' not in df.columns:
        df['missing short activities'] = False
    if 'missing secondary activities' not in df.columns:
        df['missing secondary activities'] = False

    bunching_error_cols = ["bunching labels", 'missing short activities',]
    omission_error_cols = ['missing secondary activities', 'missing posture']
    temporal_error_cols = ['overestimated duration', 'underestimated duration']

    # get the amount of bunching errors (rows where any of the bunching error cols is 'True')
    bunching_errors = df[bunching_error_cols].any(axis=1).sum()
    omission_errors = df[omission_error_cols].any(axis=1).sum()
    temporal_errors = df[temporal_error_cols].any(axis=1).sum()

    total_rows = len(df)

    total_annotations += total_rows
    total_bunching_errors += bunching_errors
    total_omission_errors += omission_errors
    total_temporal_errors += temporal_errors

print(f"Total annotations: {total_annotations}")
print(f"Total bunching errors: {total_bunching_errors} ({(total_bunching_errors/total_annotations)*100:.2f}%)")
print(f"Total omission errors: {total_omission_errors} ({(total_omission_errors/total_annotations)*100:.2f}%)")
print(f"Total temporal errors: {total_temporal_errors} ({(total_temporal_errors/total_annotations)*100:.2f}%)")
print(f"Overall error rate: {((total_bunching_errors + total_omission_errors + total_temporal_errors)/total_annotations)*100:.2f}%")