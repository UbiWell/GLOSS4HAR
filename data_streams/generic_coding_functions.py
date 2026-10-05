import sys
import os
import pytz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))

from datetime import datetime, timedelta
import agents.generic_summarizer
from data_processing.data_processing_utils import fetch_documents_between_timestamps
from data_streams.constants import GARMIN_STEPS, time_zone_dict
from agents.coding_agent import run_coding_agent
from agents.constants import USE_UEMA

coding_functions = {
    "CODING1": {
        "name": "get_results_through_data_computation",
        "description": "This function can programatically generates a python code and answer the user query using data from databases.",
        "coding_instructions": "Call this function to do perform calculation and computation. Also call this function to combine data from multiple databases.",
        "usecase": ["function_calling"],
        "function_call_instructions": "Call this function to do perform calculation and computation on wifi data.",
        "params": {
            "user_query": {"type": "str",
                           "description": "query to perform a coding and calculation task. Please include user_id (uid) in the query"},
        },
        "returns": "Provides answer to the query by performing computations data in the your database"
    }
}


class GenericCodingFunctions:
    def __init__(self, functions, req_databases):
        self.functions = functions
        self.databases = req_databases
        database_names = ', '.join(req_databases)
        self.coding_functions = {
            "CODING1": {
                "name": "get_results_through_data_computation",
                "description": f"This function can programatically generates a python code and answer the user query using data from {database_names}",
                "usecase": ["function_calling"],
                "function_call_instructions": "Call this function to do perform calculation and computation. Ask this functions to do aggregation and computation on data from multiple databases.",
                "params": {
                    "user_query": {"type": "str",
                                   "description": "the query user asked. Please include user_id in the query"},
                },
                "returns": "Provides answer to the query by performing computations data in the your database"
            }
        }

    def get_results_through_data_computation(self, user_query):
        include_statements = """
        import sys
        import os
        import warnings
        warnings.filterwarnings("ignore")
        
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agents')))
        
        
        from datetime import datetime
        from datetime import timedelta
    
        import numpy as np
        import pandas as pd
        from math import sin, cos, sqrt, atan2, radians
        
        from geopy.distance import great_circle
        from scipy import spatial
        from geopy import distance
        from shapely.geometry import MultiPoint
        from sklearn.cluster import DBSCAN
        from data_processing.data_processing_utils import fetch_documents_between_timestamps
        from data_streams.constants import *
        import folium
        from geopy.geocoders import Nominatim
        from geopy.geocoders import GoogleV3
        from agents.coding_agent import run_coding_agent 
        """
        function_imports = ""
        for database in self.databases:
            if database == "location database":
                function_imports += (
                    "\nUse following import for location functions (LOC)" + "\n" + "from data_streams.android_location import function_name")
            elif database == "uEMA database" and USE_UEMA:
                function_imports += (
                    "\nUse following import for uEMA functions (UEMA)" + "\n" + "from data_streams.pixel_uEMA import function_name")
            elif database == "heart rate database":
                function_imports += (
                    "\nUse following import for heart rate functions (HR)" + "\n" + "from data_streams.pixel_hr import function_name")
            elif database == "phone usage database":
                function_imports += (
                    "\nUse following import for phone usage functions (USAGE)" + "\n" + "from data_streams.android_phone_usage import function_name")
            elif database == "ambient noise database":
                function_imports += (
                    "\nUse following import for ambient noise functions (NOISE)" + "\n" + "from data_streams.pixel_ambient_noises import function_name")
            elif database == "step count database":
                function_imports += (
                    "\nUse following import for steps count functions (STEP)" + "\n" + "from data_streams.pixel_steps_data import function_name")
            elif database == "skin temperature database":
                function_imports += (
                    "\nUse following import for skin temperature functions (SKIN_TEMP)" + "\n" + "from data_streams.pixel_skin_temp import function_name")
            elif database == "watch wear database":
                function_imports += (
                    "\nUse following import for watch wear functions (WEAR_DETECTION)" + "\n" + "from data_streams.pixel_wear_detection import function_name")
            elif database == "wrist AUC database":
                function_imports += (
                    "\nUse following import for wrist AUC functions (WRIST_AUC)" + "\n" + "from data_streams.pixel_wrist_auc import function_name")
            # if database == "activity database":
            #     function_imports += (
            #             "\nUse following import for activity functions (ACT)" + "\n" + "from from data_streams.activity_data import function_name")

            # elif database == "location database":
            #     function_imports += (
            #             "\nUse following import for location functions (LOC)" + "\n" + "from from data_streams.location_data import function_name")

            # elif database == "phone steps database":
            #     function_imports += (
            #             "\nUse following import for phone steps functions (PHONE)" + "\n" + "from from data_streams.phone_steps_data import function_name")

            # elif database == "garmin steps database":
            #     function_imports += (
            #             "\nUse following import for garmin steps functions (GARMINSTEP)" + "\n" + "from from data_streams.garmin_steps_data import function_name")

            # elif database == "garmin hr database":
            #     function_imports += (
            #             "\nUse following import for garmin hr functions (GARMINHR)" + "\n" + "from from data_streams.garmin_hr_data import function_name")

            # elif database == "lock unlock database":
            #     function_imports += (
            #             "\nUse following import for lock unlock functions (UL)" + "\n" + "from from data_streams.lock_unlock_data import function_name")

            # elif database == "wifi database":
            #     function_imports += (
            #             "\nUse following import for wifi functions (WIFI)" + "\n" + "from from data_streams.wifi_data import function_name")

            # elif database == "app usage database":
            #     function_imports += (
            #             "\nUse following import for app usage functions (APP)" + "\n" + "from from data_streams.app_usage_data import function_name")

            # elif database == "phone battery database":
            #     function_imports += (
            #             "\nUse following import for phone battery functions (BATTERY)" + "\n" + "from from data_streams.battery_data import function_name")

            # elif database == "call log database":
            #     function_imports += (
            #             "\nUse following import for call log functions (CALL)" + "\n" + "from from data_streams.call_log import function_name")

            # elif database == "garmin stress database":
            #     function_imports += (
            #             "\nUse following import for garmin stress functions (STRESS)" + "\n" + "from from models.stress_prediction_model import function_name")

            # elif database == "brightness database":
            #     function_imports += (
            #             "\nUse following import for brightness functions (BRIGHTNESS)" + "\n" + "from from data_streams.brightness_data import function_name")

        results = run_coding_agent(user_query=user_query, database=self.databases, functions=self.functions,
                                   include_statements=include_statements, function_imports=function_imports)
        if (not results.messages):
            return "The code generation couldn't answer this query. Please try again later."
        else:
            print(results.messages)
            return results.messages[-2].content + "\n" + results.messages[-1].content.replace("TERMINATE", "")
