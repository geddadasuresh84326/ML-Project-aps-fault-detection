import pandas as pd
import numpy as np
from sensor.config import mongo_client
import pymongo
from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
import yaml
import dill
from sklearn.preprocessing import LabelEncoder

# from dotenv import load_dotenv
# load_dotenv()

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
        client = pymongo.MongoClient("mongodb+srv://mongo1:mongo@cluster0.rmfujwk.mongodb.net/?retryWrites=true&w=majority")
        logging.info(f"mongo client : {mongo_client}")
        logging.info(f"Reading data from database [{database_name}] and collection [{collection_name}]started")
        df =  pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        if "_id" in df.columns:
            logging.info("Dropping ['_id'] column")
            df = df.drop("_id",axis=1)
        logging.info(f"Reading data from database completed and the shape of the dataframe : {df.shape}")
        return df
    except Exception as e:
        raise SensorException(e, sys)
    
def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir)

        with open( file_path,"w") as file_writer:
            yaml.dump(data,file_writer)

    except Exception as e:
        raise SensorException(e, sys)

def convert_columns_float(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise SensorException(e, sys)

def target_column_encoding(target_feature):
    label_encoder = LabelEncoder()
    result_arr = label_encoder.fit_transform(target_feature)
    
    return result_arr

def save_object(file_path:str,obj:object)->None:
    try:
        logging.info(f"Enterd into save object method of utils")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info(f"Existed from save object method of utils")

    except Exception as e:
        raise SensorException(e, sys)

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file : {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise SensorException(e, sys)

def save_numpy_array_data(file_path:str,array:np.array):
    '''
    Save numpy array data to file
    file_path:str location of the file to save
    array:np array to save
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)

    except Exception as e:
        raise SensorException(e, sys)

def load_numpy_array_data(file_path:str)->np.array:
    """
    Load numpy array data from file
    file_path:str location of file
    Return : np.array of saved file
    """
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(e, sys)