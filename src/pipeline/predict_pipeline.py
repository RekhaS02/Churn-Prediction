import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred_class = model.predict(data_scaled)[0]
            pred_proba = model.predict_proba(data_scaled)[0][1]

            return pred_class, pred_proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps the raw values coming in from the Flask HTML form into a
    pandas DataFrame with the exact column names the preprocessor expects.
    """

    def __init__(self, gender, SeniorCitizen, Partner, Dependents, tenure,
                 PhoneService, MultipleLines, InternetService, OnlineSecurity,
                 OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                 StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                 MonthlyCharges, TotalCharges):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
