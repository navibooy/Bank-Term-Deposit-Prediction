import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add project root to Python path for Airflow
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DAG_ID = "training_pipeline"
DEFAULT_ARGS = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,
}

# Create DAG
dag = DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description="ML Training Pipeline",
    schedule_interval="@daily",
    max_active_runs=1,
    tags=["ml", "training", "catboost", "bank-marketing"],
)


def data_ingestion_task(**context) -> Dict[str, Any]:
    """Task 1: Data Ingestion — Downloads and loads the raw dataset."""
    try:
        logger.info("Starting data ingestion task")
        from src.data.ingest import load_dataset

        dataset = load_dataset("train")

        ingestion_result = {
            "status": "success",
            "dataset_shape": dataset.shape,
            "dataset_columns": list(dataset.columns),
            "raw_data_path": "data/raw/train.csv",
            "timestamp": datetime.now().isoformat(),
            "task": "data_ingestion",
        }

        logger.info(
            "Data ingestion completed: %s rows × %s cols",
            dataset.shape[0], dataset.shape[1],
        )
        return ingestion_result

    except Exception as error:
        logger.error("Data ingestion failed: %s", error)
        raise AirflowException(f"Data ingestion task failed: {error}")


def data_transformation_task(**context) -> Dict[str, Any]:
    """Task 2: Data Transformation — Feature engineering & train/test split."""
    try:
        logger.info("Starting data transformation task")
        ingestion_data = context["task_instance"].xcom_pull(task_ids="data_ingestion")
        logger.info("Received from ingestion: %s", ingestion_data)

        from src.data.ingest import load_dataset
        from src.features.transform import load_processed_data, transform_dataset

        try:
            X_train, X_test, y_train, y_test = load_processed_data()
            logger.info("Using cached processed data")
            status = "success_cached"
        except FileNotFoundError:
            logger.info("No cached data found — running transformation")
            raw_data = load_dataset("train")
            X_train, X_test, y_train, y_test = transform_dataset(raw_data)
            status = "success_new"

        transformation_result = {
            "status": status,
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
            "processed_data_path": "data/processed/",
            "timestamp": datetime.now().isoformat(),
            "task": "data_transformation",
        }

        logger.info("Transformation complete: train=%s, test=%s",
                    transformation_result["X_train_shape"],
                    transformation_result["X_test_shape"])
        return transformation_result

    except Exception as error:
        logger.error("Data transformation failed: %s", error)
        raise AirflowException(f"Data transformation task failed: {error}")


def model_training_task(**context) -> Dict[str, Any]:
    """Task 3: Model Training — Train CatBoost model on processed data."""
    try:
        logger.info("Starting model training task")
        transformation_data = context["task_instance"].xcom_pull(task_ids="data_transformation")
        logger.info("Received transformation data: %s",
                    transformation_data.get("status", "unknown") if transformation_data else "none")

        from src.models.train import train_catboost_model
        training_results = train_catboost_model() or {}

        model_path = (
            training_results.get("model_path")
            or training_results.get("saved_paths", {}).get("final_model_pkl")
        )

        training_result = {
            "status": training_results.get("status", "success"),
            "model_path": str(model_path) if model_path else None,
            "best_iteration": training_results.get("best_iteration"),
            "best_score": training_results.get("best_score"),
            "training_data_shape": list(training_results.get("training_data_shape", [])),
            "test_data_shape": list(training_results.get("test_data_shape", [])),
            "timestamp": datetime.now().isoformat(),
            "task": "model_training",
        }

        logger.info("Model training completed")
        logger.info("Model path: %s", training_result["model_path"])
        logger.info("Best iteration: %s", training_result["best_iteration"])
        logger.info("Best score: %s", training_result["best_score"])
        return training_result

    except Exception as error:
        logger.error("Model training failed: %s", error)
        raise AirflowException(f"Model training task failed: {error}")


def model_validation_task(**context) -> Dict[str, Any]:
    """Task 4: Model Validation — Evaluate trained model & log metrics."""
    try:
        logger.info("Starting model validation task")
        training_data = context["task_instance"].xcom_pull(task_ids="model_training") or {}
        logger.info("Received from training: %s", training_data)

        from src.models.validate import validate_model
        model_path = training_data.get("model_path")
        validation_results = validate_model(model_path=model_path)

        metrics = validation_results["metrics"]
        validation_checks = validation_results["validation_results"]

        validation_result = {
            "status": "success",
            "model_path": str(model_path),
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "validation_passed": all(validation_checks.values()),
            "validation_details": validation_checks,
            "plot_paths": {k: str(v) for k, v in validation_results["plot_paths"].items()},
            "timestamp": datetime.now().isoformat(),
            "task": "model_validation",
        }

        logger.info("Validation complete — ROC-AUC: %.5f | Passed: %s",
                    validation_result["roc_auc"],
                    validation_result["validation_passed"])
        return validation_result

    except Exception as error:
        logger.error("Model validation failed: %s", error)
        raise AirflowException(f"Model validation task failed: {error}")


def pipeline_success_callback(**context) -> None:
    """Callback when DAG succeeds — log pipeline summary."""
    try:
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        ti = context["task_instance"]
        ingestion_data = ti.xcom_pull(task_ids="data_ingestion") or {}
        transformation_data = ti.xcom_pull(task_ids="data_transformation") or {}
        training_data = ti.xcom_pull(task_ids="model_training") or {}
        validation_data = ti.xcom_pull(task_ids="model_validation") or {}

        logger.info("Pipeline Summary:")
        logger.info("  Dataset: %s", ingestion_data.get("dataset_shape"))
        logger.info("  Training set: %s", transformation_data.get("X_train_shape"))
        logger.info("  Best Iteration: %s", training_data.get("best_iteration"))
        logger.info("  ROC-AUC: %s", validation_data.get("roc_auc"))
        logger.info("  Validation: %s",
                    "✅ PASSED" if validation_data.get("validation_passed") else "❌ FAILED")
        logger.info("  Model saved at: %s", training_data.get("model_path"))
        logger.info("=" * 60)

    except Exception as error:
        logger.error("Pipeline success callback failed: %s", error)


def pipeline_failure_callback(**context) -> None:
    """Callback when DAG fails — log context & partial results."""
    try:
        logger.error("=" * 60)
        logger.error("TRAINING PIPELINE FAILED")
        logger.error("=" * 60)

        ti = context["task_instance"]
        logger.error("Failed task: %s | DAG: %s | Date: %s",
                     ti.task_id, ti.dag_id, context["ds"])

        for task_id in ["data_ingestion", "data_transformation", "model_training"]:
            task_data = ti.xcom_pull(task_ids=task_id) or {}
            logger.info("%s status: %s", task_id, task_data.get("status", "n/a"))

        logger.error("=" * 60)

    except Exception as error:
        logger.error("Pipeline failure callback failed: %s", error)


# Task definitions
data_ingestion = PythonOperator(
    task_id="data_ingestion",
    python_callable=data_ingestion_task,
    dag=dag,
)
data_transformation = PythonOperator(
    task_id="data_transformation",
    python_callable=data_transformation_task,
    dag=dag,
)
model_training = PythonOperator(
    task_id="model_training",
    python_callable=model_training_task,
    dag=dag,
)
model_validation = PythonOperator(
    task_id="model_validation",
    python_callable=model_validation_task,
    dag=dag,
)

# Dependencies
data_ingestion >> data_transformation >> model_training >> model_validation

# Callbacks
dag.on_success_callback = pipeline_success_callback
dag.on_failure_callback = pipeline_failure_callback

# Quick test entrypoint
if __name__ == "__main__":
    print(f"DAG '{DAG_ID}' loaded successfully")
    print(f"Tasks: {[task.task_id for task in dag.tasks]}")
    print("DAG is ready for Airflow deployment")