import pandas as pd
import numpy as np

PATH_TO_FOLDER = "/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/error_flagging"

# list all the files in the folder
import os
files = os.listdir(PATH_TO_FOLDER)
files = [f for f in files if f.endswith('.xlsx')]

# read each of the file into pandas dataframe
dfs = []
for file in files:
    df = pd.read_excel(os.path.join(PATH_TO_FOLDER, file))
    print(df.columns)