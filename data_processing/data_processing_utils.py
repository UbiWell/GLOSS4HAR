# from pymongo import MongoClient
from datetime import datetime
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_streams')))

from data_processing.db_config import DbConfig
import pymongo

# def fetch_documents_between_timestamps(uid, start_timestamp, end_timestamp, collection_name):
#     """
#     Fetch documents from a MongoDB collection between two timestamps.

#     Parameters:
#     - start_timestamp (datetime): The start timestamp.
#     - end_timestamp (datetime): The end timestamp.
#     - collection_name (str): The name of the collection to query.
#     - db_name (str): The name of the database. Default is 'your_database_name'.

#     Returns:
#     - list: A list of documents that match the query.
#     """
#     db = DbConfig().getDb()
#     collection = db[collection_name]
#     if collection_name == 'ios_steps':
#         query = {
#             'uid': uid,
#             'start_timestamp': {
#                 '$gte': start_timestamp,
#                 '$lt': end_timestamp
#             }
#         }
#     else:
#         query = {
#             'uid': uid,
#             'timestamp': {
#                 '$gte': start_timestamp,
#                 '$lt': end_timestamp
#             }
#         }


#     results = collection.find(query, sort=[('timestamp', pymongo.ASCENDING)])
#     documents = list(results)

#     return documents

def fetch_documents_between_timestamps(uid, start_timestamp, end_timestamp, collection_name):
    """
    Fetch documents from the csv file.

    Parameters:
    - start_timestamp (datetime): The start timestamp.
    - end_timestamp (datetime): The end timestamp.
    - collection_name (str): The name of the collection to query.
    - db_name (str): The name of the database. Default is 'your_database_name'.

    Returns:
    - list: A list of documents that match the query.
    """
    
    PATH_TO_CSV_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'convert_to_mongo_db', 'HARLLM'))

    path_to_csv_file = os.path.join(PATH_TO_CSV_FOLDER, f"{collection_name}.csv")
    # read the csv file
    df = pd.read_csv(path_to_csv_file)
    # convert the start and end timestamp to epoch (milliseconds)
    start_timestamp = int(start_timestamp * 1000)
    end_timestamp = int(end_timestamp * 1000)
    # filter the dataframe based on uid and timestamp
    filtered_df = df[(df['subject_id'] == uid) & (df['timestamp'] >= start_timestamp) & (df['timestamp'] < end_timestamp)]
    # sort the dataframe based on timestamp
    filtered_df = filtered_df.sort_values(by='timestamp')
    # divide the timestamp by 1000 to convert it to seconds
    filtered_df['timestamp'] = filtered_df['timestamp'] / 1000
    # convert the filtered dataframe to a list of dictionaries
    documents = filtered_df.to_dict('records')
    return documents


def fetch_first_and_last_document(uid, collection_name):
    """
    Fetch the first and last documents for a specific uid from a MongoDB collection.

    Parameters:
    - uid (str): The user ID to filter the documents.
    - collection_name (str): The name of the collection to query.

    Returns:
    - tuple: A tuple containing the first and last documents for the specified uid.
    """
    db = db_config.DbConfig().getDb()
    collection = db[collection_name]

    # Fetch all documents for the specified uid, sorted by timestamp
    results = collection.find({'uid': uid}).sort('timestamp', pymongo.ASCENDING)
    documents = list(results)

    # Return the first and last documents
    if documents:
        first_document = documents[0]
        last_document = documents[-1]
        return first_document, last_document
    else:
        return None, None  # Return None if no documents are found

# Example usage
if __name__ == "__main__":
    start_datetime = datetime(2025, 2, 18, 0, 0, 0)
    end_datetime = datetime(2025, 2, 20, 23, 59, 59)

    start_timestamp = start_datetime.timestamp()
    end_timestamp = end_datetime.timestamp()

    collection_name = 'android_location'  # Change this to the collection you want to query

    documents = fetch_documents_between_timestamps("pilot2", start_timestamp, end_timestamp, collection_name)

    for doc in documents:
        print(doc)

    print("total entries:", len(documents))



