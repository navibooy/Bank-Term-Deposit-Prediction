import logging
import sys
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DAG_ID = "drift_detection_pipeline"
DEFAULT_ARGS = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,
}

dag = DAG(
    DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Drift detection pipeline using Evidently + MLflow",
    schedule_interval="0 * * * *",  # run hourly
    max_active_runs=1,
    tags=["ml", "monitoring", "drift", "evidently"],
)


def drift_detection_task(**context):
    """
    Runs drift detection reports (data drift, target drift, data quality, drift tests).
    Logs results + artifacts to MLflow.
    """
    try:
        from src.monitoring.generate_drift import DriftReportGenerator

        generator = DriftReportGenerator()

        reference_path = "data/reference/reference.parquet"
        current_path = "data/current/current_batch.parquet"
        target_column = "y"

        results = generator.generate_all_reports(
            reference_path=reference_path,
            current_path=current_path,
            target_column=target_column,
        )

        drift_results = results["drift_results"]
        logger.info("Drift detection finished")
        logger.info(f"Data drift detected: {drift_results.get('data_drift_detected')}")
        logger.info(f"Target drift detected: {drift_results.get('target_drift_detected')}")

        return results

    except Exception as e:
        logger.error(f"Drift detection task failed: {e}")
        raise


# Task definition
drift_detection = PythonOperator(
    task_id="drift_detection",
    python_callable=drift_detection_task,
    dag=dag,
    doc_md="""
    ## Drift Detection Task

    Generates Evidently drift reports (data drift, target drift, data quality, drift tests).
    Logs metrics + artifacts to MLflow.
    """,
)

# DAG docs
dag.doc_md = """
# Drift Detection DAG

Monitors for data and target drift in the bank marketing prediction model.

## Pipeline Flow
1. Load reference dataset (training).
2. Load current dataset (latest batch).
3. Generate drift reports using Evidently.
4. Log drift results + artifacts to MLflow.

## Scheduling
- Runs hourly (`0 * * * *`).
- Max active runs = 1.

## Monitoring
- Drift alerts appear in Airflow logs.
- Reports logged as artifacts in MLflow.
"""

if __name__ == "__main__":
    print(f"DAG '{DAG_ID}' loaded successfully")
    print(f"Tasks: {[task.task_id for task in dag.tasks]}")