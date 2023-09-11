from sensor.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact
from sensor.entity.config_entity import DataTransformationConfig,TARGET_COLUMN

import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import target_column_encoding,save_numpy_array_data,save_object

import pandas as pd
import numpy as np 

from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig,
                      data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20}Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            pass
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy = "constant",fill_value=0)
            robust_scaler = RobustScaler()

            constant_pipeline = Pipeline(steps = [
                ('Imputer',simple_imputer),
                ('RobustScaler',robust_scaler)
            ])
            return constant_pipeline

        except Exception as e:
            raise SensorException(e, sys)
    
    def initiate_data_transformation(self)->DataTransformationArtifact:

        try:
            logging.info(f"data_transformation initiated")
            # Reading train and test dataframes
            logging.info(f"Reading train and test dataframes from data ingestion artifact")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # selecting input feature for training and testing dataframe
            logging.info(f"splitting each dataframe into input feature and target feature")
            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis = 1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis = 1)
            logging.info(f"train input feature shape: {input_feature_train_df.shape} and test input feature shape : {input_feature_test_df.shape}")
            # selecting target feature for training and testing dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info(f"train target feature shape: {target_feature_train_df.shape} and test target feature shape : {target_feature_test_df.shape}")

            # target column label encoding 
            logging.info(f"target column label encoding started")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info(f"target column label encoding completed")

            # target_feature_train_arr = target_column_encoding(target_feature_train_df)
            # target_feature_test_arr = target_column_encoding(target_feature_test_df)

            # Trasforming input feature
            logging.info(f"transforming input feature started")
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            input_feature_train_arr = transformation_pipeline.fit_transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)   
            logging.info(f"transforming input feature completed")

            logging.info(f"--- handling imbalance started using SMOTETomek -----")
            smt = SMOTETomek(random_state = 42)
            logging.info(f"Before resampling in Training set Input: {input_feature_train_arr.shape} Target : {target_feature_train_arr.shape}")
            input_feature_train_arr,target_feature_train_arr = smt.fit_resample(input_feature_train_arr,
                                                                                target_feature_train_arr)
            logging.info(f"After resampling in Training set Input: {input_feature_train_arr.shape} Target : {target_feature_train_arr.shape}")
            logging.info(f"Before resampling in Training set Input: {input_feature_test_arr.shape} Target : {target_feature_test_arr.shape}")
            
            input_feature_test_arr,target_feature_test_arr = smt.fit_resample(input_feature_test_arr,
                                                                                target_feature_test_arr)
            logging.info(f"After resampling in Training set Input: {input_feature_test_arr.shape} Target : {target_feature_test_arr.shape}")

            # concatenating input and target feature after transformation as train_arr and test_arr
            logging.info(f"After transformation combining input features and target features of train and test dataframes")
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arrr = np.c_[input_feature_test_arr,target_feature_test_arr]


            # save numpy array
            logging.info(f"saving numpy array (train_arr and test_arr)")
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arrr)

            # save transformation pipeline object
            logging.info(f"saving transformation object")
            save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            # save encoder object
            logging.info(f"saving encoder object")
            save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            # preparing artifact
            logging.info(f" preparing data transformation artifact ")
            data_transformation_artifact = DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path)
            logging.info(f"Data Transformation artifact : {data_transformation_artifact}")

            logging.info(f"data transformation completed")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)