import os

from data_processing.data_processing_utils import DATA_DIR


def _env_flag(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


# Ablations
ABLATION_PRESENTATION_AGENT = _env_flag("GLOSS4HAR_NO_PRESENTATION")
ABLATION_MEMORY = _env_flag("GLOSS4HAR_NO_MEMORY")

# Give the agents access to the uEMA self-reports (the uEMA condition of task 2)
USE_UEMA = _env_flag("GLOSS4HAR_USE_UEMA")

# Databases the agents can query. Each one is backed by a CSV in the data folder (DATABASE_FILES).
# pixel_wrist_auc.csv is in the dataset but was not offered to the agents in the paper.
databases = {
    "location database": {
        "info": "Contains GPS location data (latitude, longitude, altitude) recorded via the phone.",
        "device": "Phone",
        "additional_instructions": "The location database can be used to detected activity related to the location, such as home, work, entertainment, etc. It can also detected speed to identify activity like riding train, bus, cycling, ... The location database provides functions to calculate physical address but only call it when needed as it is computationally expensive. Do all calculation in latitude and longitude values and call this function only when you need to show the address to the user."
    },
    "uEMA database": {
        "info": "Contains user's self-report of in-the-moment activity on the watch. Some of the self-reports are voice-based, so the responses in the database are transcribed text.",
        "device": "Watch",
        "additional_instructions": "Some of the transcribed responses might not be accurate (e.g., sitting might be transcribed to setting). The uEMA database is used to retrieve what activity participant did at that moment, not the start time of the activity. To get the exact start and end time of an activity, first use information in other databases where there is a time period of consistent data, then match it with the uEMA to get the exact activity label."
    },
    "heart rate database": {
        "info": "Contains heart rate data (anytime it changes) recorded from the Pixel smartwatch.",
        "device": "Watch",
        "additional_instructions": "The heart rate database is used to understand the user's physical activity level. Combined with step count it can detect moderate to vigorous activities that doesn't involve a lot of steps."
    },
    "phone usage database": {
        "info": "Contains phone usage data. True means the phone is being used, and false means the phone is not being used at the time collected.",
        "device": "Phone",
        "additional_instructions": "The phone usage database is used to detect 'phone using' activity."
    },
    "ambient noise database": {
        "info": "Contains ambient noise data recorded from the Pixel smartwatch. The data is recorded every 5 minutes.",
        "device": "Watch",
        "additional_instructions": "The noises are detected using Google YAMNet model, which classifies the noise into 521 classes. Some of the predictions might be inaccurate or not related to the user's activity."
    },
    "step count database": {
        "info": "Contains step count data recorded from the Pixel smartwatch. The data is recorded every minute.",
        "device": "Watch",
        "additional_instructions": "The step count database is used to detect activities such as walking, running. Is used with heart rate or location, it can be used to detect activities that does not involve walking or running, such as cycling, riding car, etc.",
    },
    "skin temperature database": {
        "info": "Contains skin temperature data recorded from the Pixel smartwatch. The data is recorded every minute.",
        "device": "Watch",
        "additional_instructions": "The skin temperature database can be used to detect activities that involve changes in skin temperature, such as exercise or sleep.",
    },
    "watch wear database": {
        "info": "Contains data whether the watch is worn or not. True means the watch is worn, and false means the watch is not worn at the time collected.",
        "device": "Watch",
        "additional_instructions": "The watch wear database is used to detect activities such as wearing the watch, taking it off, or putting it on."
    }
}

DATABASE_FILES = {
    "location database": "android_location",
    "uEMA database": "uEMA",
    "heart rate database": "garmin_hr",
    "phone usage database": "android_phone_usage",
    "ambient noise database": "pixel_ambient_noise",
    "step count database": "pixel_steps",
    "skin temperature database": "pixel_skin_temperature",
    "watch wear database": "pixel_wear_detection",
}

if not USE_UEMA:
    del databases["uEMA database"]

# Drop databases whose data file is missing (e.g. android_location.csv is not in the public data release)
for _database in list(databases):
    if not os.path.exists(os.path.join(DATA_DIR, DATABASE_FILES[_database] + ".csv")):
        print(f"Warning: {DATABASE_FILES[_database]}.csv not found in {DATA_DIR}; disabling the {_database}.")
        del databases[_database]
