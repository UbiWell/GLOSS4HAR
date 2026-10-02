import pandas as pd
import pickle

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

def retrieve_GLOSS_results(subject_id, date):
    with open(f'/home/gloss/evaluation/GLOSS4HAR_results_{subject_id}_4h.pkl', 'rb') as f:
        results = pickle.load(f)

    # get all the results[hours]["answer"]
    answers = ""
    for hour in results:
        answers += f"{results[hour]['answer']}\n"

    # answers have multiple lines, each line has the format of "start time-end time: labels". split the answers into lines
    lines = answers.split("\n")
    data = []
    # extract start time, end time, and labels from each line
    for line in lines:
        if line.strip() == "":
            continue
        try:
            labels = line.split(":")[-1]
            # time range is the part before the last colon
            time_range = line.replace(":" + labels, "")
            start_time, end_time = time_range.split("-")
            labels = [label.strip() for label in labels.split(",")]
            data.append({
                "start_time": start_time.strip(),
                "end_time": end_time.strip(),
                "labels": labels
            })
        except ValueError:
            print(f"Skipping line due to unexpected format: {line}")
            continue
    # convert to dataframe
    df = pd.DataFrame(data)
    # add the date to start_time and end_time
    df['start_time'] = df['start_time'].apply(lambda x: f"{date} {x}")
    df['end_time'] = df['end_time'].apply(lambda x: f"{date} {x}")

    # convert start_time and end_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    # instead of having start and end time, make it into minute by minute. for each label, create a row for each minute in the range
    expanded_rows = []
    for _, row in df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        labels = row['labels']
        # create a row for each minute in the range
        for minute in pd.date_range(start=start_time, end=end_time, freq='T'):
            for label in labels:
                expanded_rows.append({
                    'start_time': minute,
                    'label': label.strip()
                })
    # create a new dataframe from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    # set all seconds to 0
    expanded_df['start_time'] = expanded_df['start_time'].dt.floor('min')
    # for rows with the same start_time, combine the labels into a list
    expanded_df = expanded_df.groupby('start_time')['label'].apply(lambda x: list(set(x))).reset_index()
    # rename the label column to labels
    expanded_df.rename(columns={'label': 'labels'}, inplace=True)
    return expanded_df

def retrieve_human_annotations(subject_id, date):
    # read the human annotations
    path = f"/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/annotations/{subject_id}.csv"
    df =  pd.read_csv(path)
    
    # only keep the labels, uncertainty_start, uncertainty_end
    df = df[['labels', 'uncertainty_start', 'uncertainty_end']]
    # remove the -05:00 in the uncertainty_start and uncertainty_end
    df['uncertainty_start'] = df['uncertainty_start'].str.replace('-05:00', '', regex=False)
    df['uncertainty_end'] = df['uncertainty_end'].str.replace('-05:00', '', regex=False)
    df['uncertainty_start'] = df['uncertainty_start'].str.replace('-04:00', '', regex=False)
    df['uncertainty_end'] = df['uncertainty_end'].str.replace('-04:00', '', regex=False)
    # remove the second part of the uncertainty_start and uncertainty_end
    df['uncertainty_start'] = df['uncertainty_start'].str.split('.').str[0]
    df['uncertainty_end'] = df['uncertainty_end'].str.split('.').str[0]
    # convert to datetime objects
    df['uncertainty_start'] = pd.to_datetime(df['uncertainty_start'], format='%Y-%m-%d %H:%M:%S')
    df['uncertainty_end'] = pd.to_datetime(df['uncertainty_end'], format='%Y-%m-%d %H:%M:%S')

    # instead of having start and end time, make it into minute by minute. for each label, create a row for each minute in the range
    expanded_rows = []
    for _, row in df.iterrows():
        start_time = row['uncertainty_start']
        end_time = row['uncertainty_end']
        labels = row['labels'].split(',')
        # create a row for each minute in the range
        for minute in pd.date_range(start=start_time, end=end_time, freq='T'):
            for label in labels:
                expanded_rows.append({
                    'start_time': minute,
                    'label': label.strip()
                })
    # create a new dataframe from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    # add the date to start_time and end_time
    expanded_df['start_time'] = expanded_df['start_time'].apply(lambda x: f"{date} {x.strftime('%H:%M:%S')}")
    # convert start_time and end_time to datetime
    expanded_df['start_time'] = pd.to_datetime(expanded_df['start_time'])
    # set all seconds to 0
    expanded_df['start_time'] = expanded_df['start_time'].dt.floor('min')

    # for rows with the same start_time, combine the labels into a list
    expanded_df = expanded_df.groupby('start_time')['label'].apply(lambda x: list(set(x))).reset_index()
    # rename the label column to labels
    expanded_df.rename(columns={'label': 'labels'}, inplace=True)
    return expanded_df

def match_GLOSS_human_annotations(subject_id, date):
    gloss_df = retrieve_GLOSS_results(subject_id, date)
    # print(gloss_df.head())
    human_df = retrieve_human_annotations(subject_id, date)
    # print(human_df.head())

    # merge the two dataframes on start_time, keep all rows from human_df but only matching rows from gloss_df
    merged_df = pd.merge(human_df, gloss_df, on='start_time', how='left', suffixes=('_human', '_gloss'))
    # remove rows with NaN in the labels_human column
    merged_df = merged_df.dropna(subset=['labels_human'])
    # remove rows with NaN in the labels_gloss column
    merged_df = merged_df.dropna(subset=['labels_gloss'])
    return merged_df

def calculate_accuracy(subject_id, date):
    merged_df = match_GLOSS_human_annotations(subject_id, date)
    # for each item in the list of labels_human, check if it is in the list of labels_gloss. create column 'correct' with the count of correct matches, and 'total' with the total count of label in labels_human
    merged_df['correct'] = merged_df.apply(lambda row: len(set(row['labels_human']) & set(row['labels_gloss'])), axis=1)
    merged_df['total'] = merged_df['labels_human'].apply(lambda x: len(x))
    
    # export to csv
    merged_df.to_csv(f'/mnt/study/ari_work/llm-sensemaking/convert_to_mongo_db/GLOSS4HAR_results_{subject_id}_merged.csv', index=False)

    # print the sum of correct and total
    total_correct = merged_df['correct'].sum()
    total_labels = merged_df['total'].sum()
    accuracy = total_correct / total_labels if total_labels > 0 else 0
    print(f"ABSOLUTE RESULTS: Total correct: {total_correct}, Total labels: {total_labels}, Accuracy: {accuracy:.2f}")

    # print the relative accuracy: number of rows where correct > 0 divided by the total number of rows
    relative_accuracy = (merged_df['correct'] > 0).sum() / len(merged_df) if len(merged_df) > 0 else 0
    print(f"RELATIVE RESULTS: Relative accuracy: {relative_accuracy:.3f}, std: {merged_df['correct'].std():.3f}, mean: {merged_df['correct'].mean():.3f}, median: {merged_df['correct'].median():.3f}")
    return relative_accuracy, (merged_df['correct'] > 0).sum(), len(merged_df), merged_df['correct'].std()

def calculate_overall_accuracy():
    total_correct = 0
    total_labels = 0
    accuracies = []
    for subject_id, date in participants.items():
        try:
            relative_accuracy, correct_count, total_count, std = calculate_accuracy(subject_id, date)
            print(f"Subject: {subject_id}, Date: {date}, Relative Accuracy: {relative_accuracy:.2f}, Correct Count: {correct_count}, Total Count: {total_count}, STD: {std:.3f}")
            accuracies.append(relative_accuracy)
            total_correct += correct_count
            total_labels += total_count
        except Exception as e:
            continue
    print(f"FINAL RESULTS: {total_correct=}, {total_labels=}, Overall Accuracy: {total_correct / total_labels if total_labels > 0 else 0:.2f}, STD: {pd.Series(accuracies).std():.3f}")
if __name__ == "__main__":
    calculate_overall_accuracy()
