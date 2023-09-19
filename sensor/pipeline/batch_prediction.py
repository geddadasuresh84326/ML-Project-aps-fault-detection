from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from sensor.predictor import ModelResolver
import pandas as pd
from datetime import datetime
from sensor.utils import load_object
from sensor.entity.config_entity import TARGET_COLUMN
import numpy as np

PREDICTION_DIR:str = "prediction"


def start_batch_prediction(input_file_path:str):
    try:
        logging.info(f"Batch Prediction Started")
        os.makedirs(name=PREDICTION_DIR,exist_ok=True)
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading input file path : {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace = True)

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path = model_resolver.get_latest_transformer_path())
        input_feature_names = list(transformer.feature_names_in_)
        input_arr = df[input_feature_names]
        input_arr = transformer.transform(input_arr)

        logging.info(f"Loading target encoder to encode target column")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        target_arr = df[TARGET_COLUMN]
        y_true = target_encoder.transform(target_arr)

        logging.info(f"Loading model to make predictions")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        y_pred = model.predict(input_arr)

        logging.info(f"Converting numerical predictions to categorical predictions")
        cat_y_pred = target_encoder.inverse_transform(y_pred)

        df["prediction"] = y_pred
        df["cat_prediction"] = cat_y_pred

        prediction_file_name = os.path.basename(input_file_path.replace(".csv",f"_{datetime.now().strftime('%m%d%Y_%H%M%S')}.csv"))
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index = False,header = True)

        logging.info(f"batch prediction file path : {prediction_file_path}")
        logging.info(f"Batch prediction completed")
        
        # return prediction_file_path

    except Exception as e:
        raise SensorException(e, sys)