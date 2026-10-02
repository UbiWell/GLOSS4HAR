import os
IOS_LOCATION = 'ios_location'
EMPATICA_EDA = 'empatica_eda'
IOS_EVENTS = 'ios_events'
DAILY_SUMMARY = 'daily_summaries'
GARMIN_HR = 'garmin_hr'
GARMIN_STRESS = 'garmin_stress'
EMA_RESPONSE = "ema_response"
APP_USAGE_LOGS = "app_usage_logs"
EMA_STATUS_EVENTS = "ema_status_events"

IOS_BRIGHTNESS = 'ios_brightness'
IOS_BLUETOOTH = 'ios_bluetooth'
IOS_WIFI = 'ios_wifi'
IOS_BATTERY = 'ios_battery'
IOS_LOCK_UNLOCK = 'ios_lock_unlock'
IOS_STEPS = 'ios_steps'
IOS_ACTIVITY = 'ios_activity'
IOS_ACCELEROMETER = 'ios_accelerometer'
IOS_CALLLOG = 'ios_calllog'
EMPATICA_TEMPERATURE = 'empatica_temperature'
EMPATICA_IBI = 'empatica_ibi'
EMPATICA_BATTERY = 'empatica_battery'
EMPATICA_BVP = 'empatica_bvp'
GARMIN_IBI = 'garmin_ibi'
GARMIN_RESPIRATION = 'garmin_respiration'
GARMIN_STEPS = 'garmin_steps'
GARMIN_ENERGY = 'garmin_energy'
ACCURACY = 'accuracy'

ANDROID_LOCATION = 'android_location'
ANDROID_PHONE_USAGE = 'android_phone_usage'
PIXEL_AMBIENT_NOISE = 'pixel_ambient_noise'
PIXEL_SKIN_TEMP = 'pixel_skin_temperature'
PIXEL_STEPS = 'pixel_steps'
PIXEL_WEAR_DETECTION = 'pixel_wear_detection'
PIXEL_WRIST_AUC = 'pixel_wrist_auc'
UEMA = 'uEMA'

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

home_locations = {
    # 'user_id': {'centroid': [latitude, longitude]}
}

time_zone_dict = {
    "test004": "est",
    "test006": "est",
    "test007": "est",
    "test008": "est",
    "test009": "est",
    "test010": "est",
    "test011": "est",
    "test012": "est",
    "u008": "est",
    "u009": "est",
    "u010": "est",
    "u011": "est",
    "u012": "est",
    "u013": "est",
    "u014": "est",
    "u015": "est",
    "pilot1": "est",
    "pilot2": "est",
    "pilot3": "est",
    "pilot4": "est",
    "pilot5": "est",
    "pilot6": "est",
    "pilot7": "est",
    "pilot8": "est",
    "pilot9": "est",
    "pilot10": "est",
    "pilot11": "est",
    "acai01": "est",
    "acai02": "est",
    "acai03": "est",
    "acai04": "est",
    "acai05": "est",
    "acai06": "est",
    "acai07": "est",
    "acai08": "est",
    "acai09": "est",
    "acai10": "est",
    "acai11": "est",
    "acai12": "est",
    "acai13": "est",
    "acai14": "est"
}