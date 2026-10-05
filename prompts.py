"""Query prompts sent to GLOSS4HAR.

Task 1 (annotation correction): get_correction_prompt
Task 2 (timeline generation from passive sensing + low-effort self-reports):
    - no self-report / uEMA: get_triangulation_prompt (uEMA reports are reached through the uEMA database)
    - activity list with approximate times / without times: get_list_triangulation_prompt
"""

# Output rules shared by every timeline prompt.
_TIMELINE_RULES = """        - They can do multiple activities at once. Suggest all the activities they might be doing.
        - Make sure to include posture
        - Use dynamics time periods based on changes in activities and postures.
        - Check the start and end time of each activity and posture, and make sure they are correct TO THE MINUTE.
        - If two annotations have the same posture and activity and are consecutive, merge them into one label with the start time of the first label and end time of the second label.
        - Do not output labels with overlapping time.
        """

_FULL_HOUR = "        - List activities for the entire one hour period. \n"


def get_correction_prompt(subject_id, activity, time_period, participants):
    return f"""Subject with id '{subject_id}' annotated {activity} during entire {time_period}.
Based on the data for {participants[subject_id]}, provide a more accurate annotations. A person might be doing multiple activities/postures during that time, in that case provide multiple corrected annotations.

    - A lot of the time participants under or over estimate the start and stop time. Check data 10 minutes before and after so that you can correct it.
    - Check the start and stop time of the activity and correct if needed.
    - If the activity is not correct, suggest a different activity that they were doing during that time period.
    - Respond the corrected labels in the format of a list of: [start time]-[end time]: posture: [posture]; activities: [activities]; reasoning: [reasoning].
    - Only include annotations within {time_period}, unless {activity} lasted longer.
    - If two annotations have the same posture and activity and are consecutive, merge them into one label with the start time of the first label and end time of the second label.
    - Do not output labels with overlapping time.
"""


def get_triangulation_prompt(subject_id, time_period, participants):
    return f"""On {participants[subject_id]}, can you tell me the list of postures and activities {subject_id} did from {time_period}? 

""" + _FULL_HOUR + _TIMELINE_RULES


def get_list_triangulation_prompt(subject_id, time_period, participants, activities, with_time=True):
    """Prompt for the activity-list conditions. `activities` is the participant's reported list for the day,
    with approximate times/durations (with_time=True) or only in chronological order (with_time=False)."""
    if with_time:
        reported = "reported doing the following activities (with the approximate timestamp or duration) in chronological order"
        time_rule = "        - The timestamp and duration from the timeline might not be accurate, so triangulate with the passive sensing data.\n"
    else:
        reported = "reported doing the following activities in chronological order"
        time_rule = "        - There are no timestamps, so triangulate the list with the passive sensing data and use common sense to assign time periods.\n"

    return f""" On the day {participants[subject_id]}, the subject {subject_id} {reported}: 

    ===========================
    {activities}
    ===========================
    
    Can you tell me the list of postures and activities {subject_id} did from {time_period} based on their passive sensing data an their reported timeline? 
        - Use the timeline to narrow down the activities they might be doing.
""" + time_rule + _FULL_HOUR + _TIMELINE_RULES


def get_real_time_sim_prompt(subject_id, time_period, participants, uEMA_sim, timeline_memory):
    return f"""
    On the day {participants[subject_id]}, the subject {subject_id} reported doing the following activities at these timestamp (use these to fix past mistakes):
    ===========================
    {uEMA_sim}
    ===========================

    The timeline so far (include this in final answer, with corrections and added activities):
    ===========================
    {timeline_memory}
    ===========================

    You job is to report all the activities and postures the subject did during {time_period} based on their passive sensing data, and their reported activities above.
    The memory contains the timeline of activities so far. You should 1) add new activities/postures if they are not in the timeline yet, and 2) correct any existing activities/postures in the timeline if needed.
        - Use the participants reports to fix any mistakes in the timeline and add any new activities they mentioned.
        - Update the timeline based on the new reports from participants above. Reports are fixes for the past annotations. For example if the participant said they were not "sitting" at 10:15, but the timeline has an entry: "10:00-10:15: sitting", then you should try to change the "sitting" to something else for that entry, like "10:00-10:15: standing".
        - Include the entire timeline from the memory and the new activities in the final answer.
""" + _TIMELINE_RULES
