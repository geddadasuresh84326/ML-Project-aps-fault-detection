import pymongo
import pandas as pd
import json

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb+srv://mongo1:mongo@cluster0.rmfujwk.mongodb.net/?retryWrites=true&w=majority")

DATABASE_NAME = "aps1"
COLLECTION_NAME = "sensor1"
DATA_FILE_PATH = "/config/workspace/aps_failure_training_set1.csv"

# Database Name
# dataBase = client["aps"]

# Collection  Name
# collection = dataBase['sensor']

# creating DataFrame
if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Shape of the Dataframe : {df.shape}")

    ## converting DataFrame into JSON
    df.reset_index(drop=True,inplace=True)
    json_records = list(json.loads(df.T.to_json()).values())
    print(json_records[0])

    ## insert converted json into MongohDB
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
    