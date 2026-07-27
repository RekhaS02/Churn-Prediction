import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]

NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Builds a ColumnTransformer:
          - numeric columns: median-impute missing values, then standard-scale
          - categorical columns: most-frequent-impute, then one-hot encode
        Wrapping this in a Pipeline means the EXACT same transformations
        applied at training time are guaranteed to be applied at
        prediction time - no train/serve skew.
        """
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ])

            logging.info(f"Numerical columns: {NUMERICAL_COLUMNS}")
            logging.info(f"Categorical columns: {CATEGORICAL_COLUMNS}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, NUMERICAL_COLUMNS),
                ("cat_pipeline", cat_pipeline, CATEGORICAL_COLUMNS),
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dataset-specific cleaning that has to happen before the pipeline runs."""
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
        # TotalCharges is stored as a string in the raw data and has a
        # handful of blank values for brand-new customers (tenure == 0).
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            train_df = self._clean(train_df)
            test_df = self._clean(test_df)

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr.toarray() if hasattr(input_feature_train_arr, "toarray") else input_feature_train_arr,
                               np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray() if hasattr(input_feature_test_arr, "toarray") else input_feature_test_arr,
                              np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
