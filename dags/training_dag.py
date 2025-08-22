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
    """
    Task 1: Data Ingestion
    Downloads and loads the raw dataset.

    Returns:
        Dict containing dataset info and file path via XCom
    """
    try:
        logger.info("Starting data ingestion task")

        # Import data ingestion functions
        from src.data.ingest import load_dataset

        # Load the dataset
        dataset = load_dataset("train")

        # Get dataset summary for logging
        logger.info(f"Dataset shape: {dataset.shape}")
        logger.info(f"Dataset columns: {list(dataset.columns)}")

        # Prepare XCom data (metadata only, not the full dataset)
        ingestion_result = {
            "status": "success",
            "dataset_shape": dataset.shape,
            "dataset_columns": list(dataset.columns),
            "raw_data_path": "data/raw/train.csv",  # Path where data is saved
            "timestamp": datetime.now().isoformat(),
            "task": "data_ingestion",
        }

        logger.info("Data ingestion completed successfully")
        logger.info(
            f"Dataset info: {dataset.shape[0]:,} rows × {dataset.shape[1]} columns"
        )

        return ingestion_result

    except Exception as error:
        logger.error(f"Data ingestion failed: {error}")
        raise AirflowException(f"Data ingestion task failed: {error}")


def data_transformation_task(**context) -> Dict[str, Any]:
    """
    Task 2: Data Transformation / Feature Engineering
    Transforms raw data and creates train/test splits.

    Returns:
        Dict containing transformation info via XCom
    """
    try:
        logger.info("Starting data transformation task")

        # Get ingestion results from XCom
        ingestion_data = context["task_instance"].xcom_pull(task_ids="data_ingestion")
        logger.info(f"Received from ingestion: {ingestion_data}")

        # Import transformation functions
        from src.data.ingest import load_dataset
        from src.features.transform import load_processed_data, transform_dataset

        # Check if processed data already exists
        try:
            X_train, X_test, y_train, y_test = load_processed_data()
            logger.info("Found existing processed data, using cached version")

            transformation_result = {
                "status": "success_cached",
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "y_train_shape": y_train.shape,
                "y_test_shape": y_test.shape,
                "processed_data_path": "data/processed/",
                "timestamp": datetime.now().isoformat(),
                "task": "data_transformation",
            }

        except FileNotFoundError:
            logger.info("No existing processed data found, running transformation")

            # Load raw data and transform
            raw_data = load_dataset("train")
            X_train, X_test, y_train, y_test = transform_dataset(raw_data)

            transformation_result = {
                "status": "success_new",
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "y_train_shape": y_train.shape,
                "y_test_shape": y_test.shape,
                "processed_data_path": "data/processed/",
                "features_created": ["many_no"],  # List of engineered features
                "timestamp": datetime.now().isoformat(),
                "task": "data_transformation",
            }

        logger.info("Data transformation completed successfully")
        logger.info(f"Training set: {transformation_result['X_train_shape']}")
        logger.info(f"Test set: {transformation_result['X_test_shape']}")

        return transformation_result

    except Exception as error:
        logger.error(f"Data transformation failed: {error}")
        raise AirflowException(f"Data transformation task failed: {error}")


def model_training_task(**context) -> Dict[str, Any]:
    """Task 3: Model Training - Trains CatBoost model on processed data."""
    try:
        logger.info("Starting model training task")

        # Get transformation results from previous task via XCom
        transformation_data = context["task_instance"].xcom_pull(
            task_ids="data_transformation"
        )
        if transformation_data:
            logger.info(
                f"Received transformation data: {transformation_data.get('status', 'unknown')}"
            )

        # Execute the main training pipeline
        from src.models.train import train_catboost_model

        training_results = train_catboost_model()

        # Prepare standardized XCom response aligned with train_catboost_model output
        training_result = {
            "status": training_results["status"],
            "model_path": training_results["saved_paths"]["final_model_pkl"],
            "best_iteration": training_results["best_iteration"],
            "best_score": training_results["best_score"],
            "training_data_shape": list(training_results["training_data_shape"]),
            "test_data_shape": list(training_results["test_data_shape"]),
            "timestamp": datetime.now().isoformat(),
            "task": "model_training",
        }

        logger.info("Model training task completed successfully")
        logger.info(f"Model saved to: {training_result['model_path']}")
        logger.info(f"Best iteration: {training_result['best_iteration']}")
        logger.info(f"Best score: {training_result['best_score']}")

        return training_result

    except Exception as error:
        logger.error(f"Model training task failed: {error}")
        raise AirflowException(f"Model training task failed: {error}")


def model_validation_task(**context) -> Dict[str, Any]:
    """
    Task 4: Model Validation
    Evaluates the trained model and logs metrics to MLflow.

    Returns:
        Dict containing validation results via XCom
    """
    try:
        logger.info("Starting model validation task")

        # Get training results from XCom
        training_data = context["task_instance"].xcom_pull(task_ids="model_training")
        logger.info(f"Received from training: {training_data}")

        # Import validation functions
        from src.models.validate import validate_model

        # Validate the model using the path from training results
        model_path = training_data["model_path"]
        validation_results = validate_model(model_path=model_path)

        # Extract key metrics for XCom
        metrics = validation_results["metrics"]
        validation_checks = validation_results["validation_results"]

        validation_result = {
            "status": "success",
            "model_path": model_path,
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "validation_passed": all(validation_checks.values()),
            "validation_details": validation_checks,
            "plot_paths": validation_results["plot_paths"],
            "timestamp": datetime.now().isoformat(),
            "task": "model_validation",
        }

        # Log validation summary
        logger.info("Model validation completed successfully")
        logger.info(f"ROC-AUC: {validation_result['roc_auc']:.5f}")
        logger.info(f"Validation passed: {validation_result['validation_passed']}")

        # Log warning if validation failed
        if not validation_result["validation_passed"]:
            logger.warning("Model validation checks failed!")
            for check, passed in validation_checks.items():
                if not passed:
                    logger.warning(f"Failed check: {check}")

        return validation_result

    except Exception as error:
        logger.error(f"Model validation failed: {error}")
        raise AirflowException(f"Model validation task failed: {error}")


def pipeline_success_callback(**context) -> None:
    """
    Success callback function that runs when the entire pipeline succeeds.
    Logs summary of all tasks.
    """
    try:
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        # Get results from all tasks
        ingestion_data = context["task_instance"].xcom_pull(task_ids="data_ingestion")
        transformation_data = context["task_instance"].xcom_pull(
            task_ids="data_transformation"
        )
        training_data = context["task_instance"].xcom_pull(task_ids="model_training")
        validation_data = context["task_instance"].xcom_pull(
            task_ids="model_validation"
        )

        # Log summary
        logger.info("Pipeline Summary:")
        logger.info(f"  Dataset: {ingestion_data['dataset_shape']} samples")
        logger.info(f"  Training: {transformation_data['X_train_shape']} samples")
        logger.info(f"  Model: {training_data['best_iteration']} iterations")
        logger.info(f"  ROC-AUC: {validation_data['roc_auc']:.5f}")
        logger.info(
            f"  Validation: {'✅ PASSED' if validation_data['validation_passed'] else '❌ FAILED'}"
        )
        logger.info(f"  Model saved: {training_data['model_path']}")

        logger.info("=" * 60)

    except Exception as error:
        logger.error(f"Pipeline success callback failed: {error}")


def pipeline_failure_callback(**context) -> None:
    """
    Failure callback function that runs when any task in the pipeline fails.
    Logs error information for debugging.
    """
    try:
        logger.error("=" * 60)
        logger.error("TRAINING PIPELINE FAILED")
        logger.error("=" * 60)

        # Get task instance info
        task_instance = context["task_instance"]
        logger.error(f"Failed task: {task_instance.task_id}")
        logger.error(f"DAG: {task_instance.dag_id}")
        logger.error(f"Execution date: {context['ds']}")

        # Try to get XCom data from completed tasks
        try:
            completed_tasks = [
                "data_ingestion",
                "data_transformation",
                "model_training",
            ]
            for task_id in completed_tasks:
                try:
                    task_data = context["task_instance"].xcom_pull(task_ids=task_id)
                    if task_data:
                        logger.info(f"{task_id} completed: {task_data['status']}")
                except Exception:
                    logger.info(f"{task_id}: No data available")
        except Exception as e:
            logger.error(f"Could not retrieve task information: {e}")

        logger.error("=" * 60)

    except Exception as error:
        logger.error(f"Pipeline failure callback failed: {error}")


# Define tasks
data_ingestion = PythonOperator(
    task_id="data_ingestion",
    python_callable=data_ingestion_task,
    dag=dag,
    doc_md="""
    ## Data Ingestion Task

    Downloads and loads the raw bank marketing dataset.

    **Outputs to XCom:**
    - Dataset shape and column information
    - File path to raw data
    - Execution timestamp
    """,
)

data_transformation = PythonOperator(
    task_id="data_transformation",
    python_callable=data_transformation_task,
    dag=dag,
    doc_md="""
    ## Data Transformation Task

    Processes raw data and creates train/test splits with feature engineering.

    **Inputs from XCom:**
    - Dataset information from data_ingestion

    **Outputs to XCom:**
    - Processed dataset shapes
    - Feature engineering details
    - File paths to processed data
    """,
)

model_training = PythonOperator(
    task_id="model_training",
    python_callable=model_training_task,
    dag=dag,
    doc_md="""
    ## Model Training Task

    Trains CatBoost classifier on processed data.

    **Inputs from XCom:**
    - Processed data information from data_transformation

    **Outputs to XCom:**
    - Model file paths
    - Training metrics (best iteration, score)
    - Model metadata
    """,
)

model_validation = PythonOperator(
    task_id="model_validation",
    python_callable=model_validation_task,
    dag=dag,
    doc_md="""
    ## Model Validation Task

    Evaluates trained model and logs metrics to MLflow.

    **Inputs from XCom:**
    - Model information from model_training

    **Outputs to XCom:**
    - Validation metrics (ROC-AUC, Precision, etc.)
    - Validation check results
    - Plot file paths
    """,
)

# Set task dependencies
data_ingestion >> data_transformation >> model_training >> model_validation

# Add success/failure callbacks to the DAG
dag.on_success_callback = pipeline_success_callback
dag.on_failure_callback = pipeline_failure_callback

# Optional: Add task-level callbacks for individual task monitoring
data_ingestion.on_failure_callback = lambda context: logger.error(
    "Data ingestion task failed"
)
data_transformation.on_failure_callback = lambda context: logger.error(
    "Data transformation task failed"
)
model_training.on_failure_callback = lambda context: logger.error(
    "Model training task failed"
)
model_validation.on_failure_callback = lambda context: logger.error(
    "Model validation task failed"
)

# DAG documentation
dag.doc_md = """
# ML Training Pipeline DAG

This DAG orchestrates the complete machine learning training pipeline for the bank marketing prediction model.

## Pipeline Flow

1. **Data Ingestion** → Downloads and loads raw dataset
2. **Data Transformation** → Feature engineering and train/test split
3. **Model Training** → Trains CatBoost classifier
4. **Model Validation** → Evaluates model and logs to MLflow

## XCom Communication

Tasks communicate through Airflow XCom to pass metadata and file paths:
- Each task returns a dictionary with status, paths, and key metrics
- Downstream tasks pull required information from upstream tasks
- Full datasets are stored in files, not XCom (for performance)

## Scheduling

- **Schedule**: Daily (`@daily`)
- **Start Date**: 1 day ago
- **Catchup**: Disabled
- **Max Active Runs**: 1 (prevents overlapping runs)

## Monitoring

- Success/failure callbacks log pipeline summaries
- Individual task failures are logged with context
- All metrics and artifacts logged to MLflow for tracking

## Configuration

Pipeline behavior controlled by `config.yaml`:
- Model hyperparameters
- Validation thresholds
- MLflow settings
- File paths

## Usage

Deploy this DAG to your Airflow instance and it will automatically:
- Run daily training pipelines
- Monitor model performance
- Log all artifacts to MLflow
- Alert on validation failures
"""

if __name__ == "__main__":
    # This allows testing the DAG file for syntax errors
    print(f"DAG '{DAG_ID}' loaded successfully")
    print(f"Tasks: {[task.task_id for task in dag.tasks]}")
    print("DAG is ready for Airflow deployment")
