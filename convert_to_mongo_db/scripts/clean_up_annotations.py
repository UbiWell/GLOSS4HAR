import pandas as pd
from bson import ObjectId
import time
from datetime import datetime
import sys
import os
from tqdm import tqdm  # Add tqdm import
from pymongo.errors import BulkWriteError
from dateutil import parser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data_streams')))

from data_processing.db_config import DbConfig

path_to_annotation_folder = "/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations"

def clean_up_annotation_from_file(csv_file):
    df = pd.read_csv(csv_file)

    # only keep subject, date, labels, uncertainty_start and uncertainty_end columns
    df = df[['subject', 'date', 'labels', 'uncertainty_start', 'uncertainty_end']]
    # remove rows where labels is NaN
    df = df.dropna(subset=['labels'])

    # only keep the date in YYYY-MM-DD format in the date column
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # only keep hour and minute in format HH:MM in uncertainty_start and uncertainty_end columns
    df['uncertainty_start'] = pd.to_datetime(df['uncertainty_start'], format='mixed', errors='coerce').dt.strftime('%H:%M')
    df['uncertainty_end'] = pd.to_datetime(df['uncertainty_end'], format='mixed', errors='coerce').dt.strftime('%H:%M')
    
    # combine labels in rows with same uncertainty_start and uncertainty_end, separated by comma
    df = df.groupby(['subject', 'date', 'uncertainty_start', 'uncertainty_end'], as_index=False).agg({'labels': lambda x: ', '.join(x)})
    
    # save the cleaned dataframe to a new csv file
    cleaned_csv_file = csv_file.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_csv_file, index=False)
    return

# list all csv files in the annotation folder
csv_files = [f for f in os.listdir(path_to_annotation_folder) if f.endswith('.csv')]
for csv_file in csv_files:
    csv_file_path = os.path.join(path_to_annotation_folder, csv_file)
    print(f"Processing file: {csv_file_path}")
    clean_up_annotation_from_file(csv_file_path)