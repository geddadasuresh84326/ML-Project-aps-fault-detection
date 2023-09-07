import pandas as pd
from sensor.config import mongo_client
from sensor.logger import logging
from sensor.exception import SensorException
import os,sys

# function to get the data from mongoDB

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    '''
    Description:This function returns collection as DataFrame
    =========================================================
    Params:
    database_name: requires database name
    collection_name: requires collection name
    =========================================================
    Returns:
    Pandas DataFrame of a collection
    '''
    try:
        logging.info(f"Reading data from database [{database_name}] and collection [{collection_name}]started")
        df =  pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        if "_id" in df.columns:
            logging.info("Dropping ['_id'] column")
            df = df.drop("_id",axis=1)
        logging.info(f"Reading data from database completed and the shape of the dataframe : {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e, sys)
    
    