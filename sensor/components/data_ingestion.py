from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor import utils
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config= data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            logging.info(f"Data ingestion started")

            # Exporting collection as DataFrame
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"DataFrame Fetched from database and it shape is {df.shape}")
            # Save DataFrame in Feature store
            df.replace(to_replace="na",value=np.NAN,inplace= True)

            # Create feature store folder
            feature_store_file_path =os.path.dirname( self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_file_path,exist_ok=True)

            # Save df to feature store
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            logging.info(f"dataframe stored in feature store")

            # split the data into train and test DataFrames
            train_df,test_df = train_test_split(df,test_size = self.data_ingestion_config.test_size,random_state=42)

            # Create dataset directory to save train and test dataframes if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            # saving dataframes into dataset folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index = False,header = True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index = False,header = True)
            logging.info(f"dataframe stored in train_file_path and test_file_path")

            # Prepare artifact
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path, 
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path= self.data_ingestion_config.test_file_path)
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
    