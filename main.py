from sensor.exception import SensorException
from sensor.logger import logging
import sys,os
from sensor.utils import get_collection_as_dataframe
from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation

try:
     # logging.info("addition started")
     # print(1/0)
     # logging.info("addition completed")
     # get_collection_as_dataframe(database_name="aps", collection_name="sensor")
     training_pipeline_config = TrainingPipelineConfig()
     data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
     data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
     data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

     # data validaiton
     data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
     data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
     data_validation_artifact = data_validation.initiate_data_validation()

     # data transformation 
     data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
     data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact) 
     data_transformation_artifact = data_transformation.initiate_data_transformation()

except Exception as e:
     logging.info(e)
     raise SensorException(e, sys)