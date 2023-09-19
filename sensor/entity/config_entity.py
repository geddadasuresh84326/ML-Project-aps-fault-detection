import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from datetime import datetime

FILE_NAME:str = "sensor.csv"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
TARGET_COLUMN = 'class'
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = 'model.pkl'

class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir:str = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y_%H%M%S')}")


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.database_name:str = "aps"
        self.collection_name:str = "sensor"
        self.data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
        self.feature_store_file_path:str = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
        self.train_file_path:str = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
        self.test_file_path:str = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
        self.test_size:float = 0.2
        
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
        self.report_file_path:str = os.path.join(self.data_validation_dir,"report.yaml")
        self.missing_threshold:float =  0.2
        self.base_df_file_path:str = os.path.join("aps_failure_training_set1.csv")

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,"data_transformation")
        self.transform_object_path:str = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path:str = os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transformed_test_path:str = os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv", "npz"))
        self.target_encoder_path:str = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)

class ModelTrainerConfig:
        def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.model_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir,"model_trainer")
            self.model_path:str = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
            self.expected_score:float = 0.7
            self.overfitting_threshold:float = 0.1
            
class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.01

class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir:str = os.path.join(training_pipeline_config.artifact_dir,"model_pusher")
        self.saved_models_dir:str = os.path.join("saved_models")
        self.pusher_model_dir:str = os.path.join(self.model_pusher_dir,"saved_models")
        
        self.pusher_model_path:str = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path:str = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_target_encoder_path:str = os.path.join(self.pusher_model_dir,TARGET_ENCODER_OBJECT_FILE_NAME)

