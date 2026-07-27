import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    """
    This is the single entry point that ties every component together:
    ingest -> transform -> train. Running `python -m src.pipeline.train_pipeline`
    reproduces the whole model-building process from scratch.
    """
    try:
        logging.info(">>> Training pipeline started <<<")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

        trainer = ModelTrainer()
        best_model_name, report = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(">>> Training pipeline completed <<<")

        print(f"\nBest model: {best_model_name}")
        for name, metrics in report.items():
            print(f"  {name}: F1={metrics['f1_score']:.4f}  Acc={metrics['accuracy']:.4f}  AUC={metrics['roc_auc']}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
