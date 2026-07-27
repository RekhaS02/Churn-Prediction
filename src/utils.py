import os
import sys
import pickle

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """Pickle any Python object (model, preprocessor) to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a pickled object back from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Runs GridSearchCV for every candidate model, fits the best combination
    of hyperparameters, and returns a report of test-set scores so we can
    pick the single best model in model_trainer.py.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_f1 = f1_score(y_test, y_test_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_proba)
            except AttributeError:
                test_auc = None

            report[model_name] = {
                "f1_score": test_f1,
                "accuracy": test_acc,
                "roc_auc": test_auc,
                "best_params": gs.best_params_,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
