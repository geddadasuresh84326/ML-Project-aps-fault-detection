from sensor.exception import SensorException
from sensor.logger import logging
import sys,os
from sensor.utils import get_collection_as_dataframe
from sensor.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher


def start_training_pipeline():

    try:
        training_pipeline_config = TrainingPipelineConfig()
        
        # data ingestion
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

        # model trainer
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact = data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        # model evaluation
        model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(data_ingestion_artifact = data_ingestion_artifact, 
                                            model_evaluation_config = model_evaluation_config, 
                                            data_transformation_artifact = data_transformation_artifact, 
                                            model_trainer_artifact = model_trainer_artifact)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
        
        # model pusher
        model_pusher_config = ModelPusherConfig(training_pipeline_config= training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                    model_trainer_artifact = model_trainer_artifact,
                                    data_transformation_artifact = data_transformation_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()
        
    except Exception as e:
        logging.info(e)
        raise SensorException(e, sys)