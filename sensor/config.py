import pymongo
import pandas
import json
from dataclasses import dataclass
import os

# from dotenv import load_dotenv
# load_dotenv()

@dataclass
class EnviromentVariable():
    mongodb_url:str = os.getenv("MONGO_DB_URL")

env_vrble = EnviromentVariable()
mongo_client = pymongo.MongoClient(env_vrble.mongodb_url)

TARGET_COLUMN_MAPPING = {
    "pos":1,
    "neg":0
}
