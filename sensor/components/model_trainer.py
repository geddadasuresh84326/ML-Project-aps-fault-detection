from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.entity.config_entity import ModelTrainerConfig,TARGET_COLUMN

import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_numpy_array_data,save_object

import pandas as pd
import numpy as np 

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                      data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
    
    def fine_tune(self):
        try:
            # Write code for GridSearchCV
            ...
        except Exception as e:
            raise SensorException(e, sys)

    
    def train_model(self,X,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(X,y)

            return xgb_clf

        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Model trainer is initiated")
            logging.info(f"Loading train and test array")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            # splitting data into X_train, y_train and X_test,y_test
            logging.info(f"Splitting input and target features from both train and test array")
            X_train, y_train = train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            # model training
            logging.info(f"Train our model")
            model = self.train_model(X=X_train, y=y_train)

            # model prediction for X_train and X_test
            logging.info(f"Predicting with trained model")
            yhat_train = model.predict(X_train)
            yhat_test = model.predict(X_test)

            # model evaluation
            f1_train_score = f1_score(y_true = y_train,y_pred = yhat_train)
            f1_test_score = f1_score(y_true = y_test,y_pred = yhat_test)
            logging.info(f"Evaluating the model f1_train_score : {f1_train_score} and f1_test_score : {f1_test_score}")

            # check for overfitting or underfitting or expected score
            logging.info(f"checking if our model is underfitting or  not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good, as it is not able to give expected accuracy : \
                 {self.model_trainer_config.expected_score} ,model actual score : {f1_test_score}")

            logging.info(f"checking if our model is overfitting or  not")
            diff = abs(f1_train_score-f1_test_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test score diff {diff} is more than Overfitting_threshold {self.model_trainer_config.overfitting_threshold}")
            
            # save trained model
            logging.info(f"saving the trained model")
            save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # prepare artifact
            logging.info(f"Preparing model trainer artifact")
            model_trainer_artifact = ModelTrainerArtifact(
                model_path = self.model_trainer_config.model_path, 
                f1_train_score = f1_train_score, 
                f1_test_score = f1_test_score)

            logging.info(f"Model trainer artifact : {model_trainer_artifact}")
            logging.info(f"Model trainer phase completed")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)