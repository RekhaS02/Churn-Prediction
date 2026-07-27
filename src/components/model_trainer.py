import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
                "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Logistic Regression": {"C": [0.01, 0.1, 1, 5, 10],"penalty": ["l1", "l2"],"solver": ["liblinear", "saga"]},
                "Decision Tree": {"max_depth": [3, 5, 10, None]},
                "Random Forest": {"n_estimators": [100, 200, 300],"max_depth": [8, 12, 16, None],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4],"max_features": ["sqrt", "log2"]},
                "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Pick the model with the best F1 score (better than accuracy for
            # churn since the classes are imbalanced - ~73% stay, ~27% churn).
            best_model_name = max(report, key=lambda name: report[name]["f1_score"])
            best_model_score = report[best_model_name]["f1_score"]

            logging.info(f"Best model: {best_model_name} with F1 score {best_model_score:.4f}")

            if best_model_score < 0.5:
                raise CustomException("No model achieved an acceptable F1 score", sys)

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_name, report

        except Exception as e:
            raise CustomException(e, sys)