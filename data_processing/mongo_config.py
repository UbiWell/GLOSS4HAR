import os

host = os.getenv('MONGO_HOST', 'localhost')
port = os.getenv('MONGO_PORT', '27017')
database = os.getenv('MONGO_DB', 'gloss')
username = os.getenv('MONGO_USER', '')
password = os.getenv('MONGO_PASSWORD', '')
