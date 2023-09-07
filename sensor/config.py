import pymongo
import pandas
import json
from dataclasses import dataclass
import os

@dataclass
class EnviromentVariable():
    mongodb_url:str = os.getenv("MONGO_DB_URL")


env_vrble = EnviromentVariable()
mongo_client = pymongo.MongoClient(env_vrble.mongodb_url)

