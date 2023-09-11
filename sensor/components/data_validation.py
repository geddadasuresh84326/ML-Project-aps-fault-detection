from sensor.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from sensor.entity.config_entity import DataValidationConfig
import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
import pandas as pd
from typing import Optional
from sensor.utils import write_yaml_file,convert_columns_float
import numpy as np 
from sensor.entity.config_entity import TARGET_COLUMN

class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
            logging.info(f"{'>>'*20}Data Validation {'<<'*20}")
        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            base_df_columns = base_df.columns
            current_df_columns = current_df.columns
            missing_columns = []
            for base_column in base_df_columns:
                if base_column not in current_df_columns:
                    missing_columns.append(base_column)
            if len(missing_columns)>0:
                self.validation_error[f"{report_key_name} missing columns"] = missing_columns
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)
    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """

        This function will drop column which has missing values more than specified threshold

        df:accepts a DataFrame
        threshold:Percetange criteria to drop column
        =====================================================================================
        Returns: Pandas DataFrame if atleast single column available after missing have been dropped else None
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            # selecting columns which containes null values more than 30%
            drop_columns = null_report[null_report>threshold].index
            # dropping columns which have missing values
            logging.info(f"{report_key_name} dropped_columns : {list(drop_columns)}")
            logging.info(f"dataframe shape : {df.shape}")
            logging.info(f"dataframe 1 st row : {df.head()}")
            self.validation_error[f"{report_key_name} dropped_columns"] = list(drop_columns)
            
            df.drop(list(drop_columns),axis=1,inplace=True)

            # return None if no columns exist
            if len(df.columns) == 0:
                return None
            return df
            
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report = dict()
            base_df_columns = base_df.columns
            current_df_columns = current_df.columns
            for base_column in base_df_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                same_distribution = ks_2samp(base_data,current_data)

                if same_distribution.pvalue>0.05:
                    # same distribution 
                    drift_report[base_column] = {
                        "p_value " :float(same_distribution.pvalue),
                        "is_same_distribution":True
                    }
                else:
                    # different distribution
                    drift_report[base_column] = {
                        "p_value " :float(same_distribution.pvalue),
                        "is_same_distribution":False
                    }
            self.validation_error[f"{report_key_name} drift report"] = drift_report

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info(f"loading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_df_file_path)
            base_df.replace({"na":np.NAN},inplace = True)

            logging.info(f"dropping missing values columns in base_df")
            
            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="base_df")

            logging.info(f"reading train dataframe")
            logging.info(f"train file path {self.data_ingestion_artifact.train_file_path}")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"dropping missing values columns in train_df")
            train_df = self.drop_missing_values_columns(df=train_df,report_key_name="train_df")
            logging.info(f"dropping missing values columns in test_df")
            test_df = self.drop_missing_values_columns(df=test_df,report_key_name="test_df")
            
            exclude_columns = [TARGET_COLUMN]
            base_df = convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            logging.info(f"checking if the required columns exist or not")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df,report_key_name="train_df")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df,report_key_name="test_df")

            if train_df_columns_status:
                logging.info(f"as all the columns existed in train dataframe,checking data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="train_df")
            if test_df_columns_status:
                logging.info(f"as all the columns existed in test dataframe,checking data drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="test_df")

            logging.info(f"writing the report into report.yaml file")
            write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = DataValidationArtifact(report_file_path = self.data_validation_config.report_file_path) 
            logging.info(f"data_validation_artifact : {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)