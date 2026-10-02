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

# Load CSV
df = pd.read_csv("/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/HARLLM/wear_detection_data.csv")

# count the number of nan values in the df
nan_count = df.isna().sum().sum()
# drop rows with nan values
df = df.dropna()
event_id = 442
device_id = "android"

def to_epoch(ts):
    dt = parser.isoparse(ts)
    return int(dt.timestamp())

mongo_docs = []
db = DbConfig().getDb()
collection = db['pixel_wear_detection']

# Use tqdm to show progress
for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    try: 
        data = row["wear_detection"]
        # longitude = float(row["longitude"])
        # latitude = float(row["latitude"])
    except ValueError:
        # print(f"Skipping row {i} due to invalid value: {row['longitude']}, {row['latitude']}")
        continue
    uid = row['subject_id']
    doc = {
        "_id": ObjectId(),
        "uid": uid,
        "event_id": event_id,
        "timestamp": int(row["timestamp"]/1000),
        "wear_detection": data,
        "device": device_id,
        "status": "locked"
    }
    mongo_docs.append(doc)

# print the document count
print(f"Total documents to insert: {len(mongo_docs)}")
# Insert to MongoDB
if mongo_docs:
    try:
        collection.insert_many(mongo_docs, ordered=False)
    except BulkWriteError as e:
        print(f"Some documents were not inserted due to duplication")
