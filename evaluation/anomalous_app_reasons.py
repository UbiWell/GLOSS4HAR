import pandas as pd
from datetime import datetime
import sensemaking_process

user = "test008"
uids = ["test006", "test007", "test008", "test009"]


def generate_prompt(user, date):
    prompt = f"{user} reported anomalous phone app usage on {date} compared to other days. Can you compare aggregated daily app usage for last 7 days and identify the reasons or patterns for anomalous use on this particular day {date}?"
    return prompt


def generate_anomalous_dates():
    only_daily = True
    both_true = []
    only_detected = []
    only_reported = []
    both_false = []

    for uid in uids:
        anomaly = pd.read_csv(f"anomalous_use_case/{uid}_sensemaking_use_case_50.csv")
        for i, row in anomaly.iterrows():
            date = row["days"]
            is_detected_anomaly = row["anomaly"]
            if (only_daily):
                is_reported_anomaly = 1 if 'daily' in row["self_report_anomaly"] else 0
            else:
                is_reported_anomaly = 1 if len(row["self_report_anomaly"]) > 0 else 0

            if is_detected_anomaly and is_reported_anomaly:
                both_true.append((uid, date))
            elif is_detected_anomaly and not is_reported_anomaly:
                only_detected.append((uid, date))
            elif not is_detected_anomaly and is_reported_anomaly:
                only_reported.append((uid, date))
            else:
                both_false.append((uid, date))

    both_true_df = pd.DataFrame(both_true, columns=["uid", "date"])
    only_detected_df = pd.DataFrame(only_detected, columns=["uid", "date"])
    only_reported_df = pd.DataFrame(only_reported, columns=["uid", "date"])
    both_false_df = pd.DataFrame(both_false, columns=["uid", "date"])

    both_true_df.to_csv("both_true_50.csv", index=False)
    only_detected_df.to_csv("only_detected_50.csv", index=False)
    only_reported_df.to_csv("only_reported_50.csv", index=False)
    both_false_df.to_csv("both_false_50.csv", index=False)

    both_true_counts = both_true_df.groupby("uid").size().reset_index(name="count")
    only_detected_counts = only_detected_df.groupby("uid").size().reset_index(name="count")
    only_reported_counts = only_reported_df.groupby("uid").size().reset_index(name="count")
    both_false_counts = both_false_df.groupby("uid").size().reset_index(name="count")

    # Get all unique UIDs from all categories
    all_uids = set(both_true_counts["uid"]).union(
        only_detected_counts["uid"],
        only_reported_counts["uid"],
        both_false_counts["uid"]
    )

    # Print counts for each UID
    for uid in sorted(all_uids):
        both_true_count = both_true_counts[both_true_counts["uid"] == uid]["count"].sum()
        only_detected_count = only_detected_counts[only_detected_counts["uid"] == uid]["count"].sum()
        only_reported_count = only_reported_counts[only_reported_counts["uid"] == uid]["count"].sum()
        both_false_count = both_false_counts[both_false_counts["uid"] == uid]["count"].sum()

        print(f"UID: {uid}")
        print(f"  both true: {both_true_count}")
        print(f"  only detected: {only_detected_count}")
        print(f"  only reported: {only_reported_count}")
        print(f"  both false: {both_false_count}")
        print()


# generate_anomalous_dates()















# print(prompt)
# print()
# query = row["query"]
# presentation_instructions = row["presentation_instructions"]
# print(f"Query: {query}")
# print(f"Presentation Instructions: {presentation_instructions}")
# print()
# sensemaker = sensemaking_process.SenseMaker(
#     query,
#     presentation_instructions
# )
# sensemaker.make_sense()
# answer = sensemaker.answer
# print(f"Answer: {answer}")
#








# break
# sensemaker = sensemaking_process.SenseMaker(
#     query,
#     presentation_instructions
# )
# sensemaker.make_sense()
# answer = sensemaker.answer
# print(f"Answer: {answer}")
# break
