database_information = {
    "activity database": {
        "info": "Contains Activity data between two given time periods. Activities are “stationary”, “automotive”, “cycling”, “walking”, and “running”.",
        "data_records": "[{'_id': ObjectId('66a27ec479ebf8b7dc751781'), 'uid': 'test004', 'timestamp': 1721448908.0, 'activity': ['stationary'], 'confidence': 'high'}, {'_id': ObjectId('66a27ec879ebf8b7dc79bc00'), 'uid': 'test004', 'timestamp': 1721456175.0, 'activity': ['stationary'], 'confidence': 'high'}]"
    },
    "location database": {
        "info": "Contains latitude, longitude, and altitude data between two given time periods per minute",
        "data_records": "[{'_id': ObjectId('66958b795e17b58f4ebfedd0'), 'uid': 'test004', 'timestamp': 1720497708.0, 'event_id': 151, 'latitude': 40.712776, 'longitude': -74.005974, 'accuracy': 5.0, 'altitude': 21.663588}, {'_id': ObjectId('66958b795e17b58f4ebfedd1'), 'uid': 'test004', 'timestamp': 1720497830.0, 'event_id': 151, 'latitude': 40.712776, 'longitude': -74.005974, 'accuracy': 5.0, 'altitude': 21.663588}, {'_id': ObjectId('66958b795e17b58f4ebfedd2'), 'uid': 'test004', 'timestamp': 1720497953.0, 'event_id': 151, 'latitude': 40.712776, 'longitude': -74.005974, 'accuracy': 5.0, 'altitude': 21.663588}]"
    }
}


def generate_database_prompt(given_question):
    prompt = "We have following databases in our system:\n"
    for key, value in database_information.items():
        prompt = "{}\nDatabase: {}\nInfo: {}\nSample Data Records: {}\n".format(prompt, key, value["info"], value["data_records"])

    prompt = prompt + "Can you answer the given question using data from above databases? Just return yes/no answer"
    prompt = "{}\n\n{}".format(prompt, given_question)
    return prompt

print(generate_database_prompt("Can I know which location I was cycling on 2024-07-20?"))





