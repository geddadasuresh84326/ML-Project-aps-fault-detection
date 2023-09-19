from sensor.predictor import ModelResolver
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.entity.artifact_entity import DataIngestionArtifact,ModelEvaluationArtifact,DataTransformationArtifact,ModelTrainerArtifact
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from sensor.utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
from sensor.entity.config_entity import TARGET_COLUMN

class ModelEvaluation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                      model_evaluation_config:ModelEvaluationConfig,
                      data_transformation_artifact:DataTransformationArtifact,
                      model_trainer_artifact:ModelTrainerArtifact
                                          ):
        try: 
            logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            logging.info(f"Initiated model evaluation")
            ## if save_models folder has model then we will compare trained model and model in save_models folders(production)
            
            # 1.At initial stage there is no saved model
            latest_dir = self.model_resolver.get_latest_dir_path()
            if latest_dir is None:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted = True,
                    improved_accuracy = None)
                logging.info(f"Model Evaluation artifact : {model_evaluation_artifact}")
                return model_evaluation_artifact

            # 2. If there is a model inside saved models we compare
            # finding locations of model,transformer,target_encoder paths
            logging.info(f"finding locations of model,transformer,target_encoder paths")
            model_path = self.model_resolver.get_latest_model_path()
            transformer_path = self.model_resolver.get_latest_transformer_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            # Previously trained objects
            logging.info(f"Previously trained objects")
            model = load_object(file_path=model_path)
            transformer = load_object(file_path=transformer_path)
            target_encoder = load_object(file_path=target_encoder_path) 

            # Currently trained model objects
            logging.info(f"Currently trained model objects")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # comparing these models with testing dataset
            logging.info(f"comparing these models with testing dataset")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            # accuracy using previously trained model
            y_true = target_encoder.transform(target_df)
            input_feature_names = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_names])
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model : {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true = y_true,y_pred = y_pred)
            logging.info(f"Previous model f1 score : {previous_model_score}")

            # accuracy using current trained model
            input_arr = current_transformer.transform(test_df[input_feature_names])
            y_true = current_target_encoder.transform(target_df)
            y_pred = current_model.predict(input_arr)
            print(f"Prediction using current model : {target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true = y_true,y_pred = y_pred)
            logging.info(f"Current model f1 score : {current_model_score}")
            if current_model_score <= previous_model_score:
                raise Exception(f"Current trained model  is not better than previous model")
            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=current_model_score - previous_model_score)
            logging.info(f"Model eval artifact : {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)