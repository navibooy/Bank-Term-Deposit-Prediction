"""
MLOps Deployment DAG for Bank Term Deposit Prediction

This DAG handles the deployment pipeline for the bank term deposit prediction model:
- Waits for successful completion of the training pipeline
- Fetches the latest candidate model and its validation metrics
- Evaluates model against configured thresholds and promotes to MLflow Model Registry
- Triggers FastAPI service reload to serve the new champion model

The DAG is triggered by sensor (schedule=None) and waits for the training_pipeline
to complete successfully before beginning the deployment evaluation process.

Inputs:
- Latest model metrics from training_pipeline via MLflow
- Promotion thresholds from config.yaml (defaults: roc_auc >= 0.85, f1_score >= 0.75)

Actions:
- Model evaluation against performance thresholds
- MLflow Model Registry promotion to 'Production' stage
- FastAPI service reload via HTTP endpoint or container restart

Trigger: ExternalTaskSensor waiting for training_pipeline.model_validation task
"""

import logging
import os
import subprocess
from datetime import timedelta
from typing import Any, Dict

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_DATE = pendulum.datetime(2025, 8, 27, tz="Asia/Manila")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="deployment_pipeline",
    default_args=default_args,
    description="MLOps deployment pipeline for model promotion and API reload",
    schedule=None,
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    tags=["ml", "deployment", "mlflow", "registry"],
    doc_md=__doc__,
)


def fetch_latest_candidate(**context) -> Dict[str, Any]:
    """
    Fetch the latest candidate model from MLflow for promotion evaluation.
    
    Returns:
        Dict containing run_id, model_uri, metrics, and metadata
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    logger.info("Fetching latest candidate model from MLflow")
    
    try:
        client = MlflowClient()
        
        # Get the experiment (assuming default experiment name from config)
        experiment_name = "bank-marketing-catboost"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        except AttributeError:
            # Fallback to default experiment
            experiment_id = "0"
            logger.warning(f"Experiment '{experiment_name}' not found, using default experiment")
        
        # Get the latest successful run
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["end_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise ValueError("No completed runs found in MLflow")
        
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        # Extract metrics
        metrics = {
            "roc_auc": latest_run.data.metrics.get("roc_auc", 0.0),
            "f1_score": latest_run.data.metrics.get("f1_score", 0.0),
            "precision": latest_run.data.metrics.get("precision", 0.0),
            "recall": latest_run.data.metrics.get("recall", 0.0),
        }
        
        model_uri = f"runs:/{run_id}/model"
        
        result = {
            "run_id": run_id,
            "model_uri": model_uri,
            "metrics": metrics,
            "timestamp": pendulum.now().isoformat(),
            "task": "fetch_latest_candidate"
        }
        
        logger.info(f"Found candidate model: {run_id} with ROC-AUC: {metrics['roc_auc']:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch latest candidate: {str(e)}")
        raise


def promote_if_approved(**context) -> Dict[str, Any]:
    """
    Evaluate model against thresholds and promote to MLflow Model Registry if approved.
    
    Returns:
        Dict containing promotion status, model details, and reason
    """
    import yaml
    from src.deployment.promote import promote_to_champion
    
    logger.info("Evaluating model for promotion")
    
    # Pull candidate info from previous task
    candidate_info = context["task_instance"].xcom_pull(task_ids="fetch_latest_candidate")
    
    if not candidate_info:
        raise ValueError("No candidate information received from fetch_latest_candidate task")
    
    # Load thresholds from config
    try:
        with open("/opt/airflow/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        thresholds = config.get("model", {}).get("promotion_thresholds", {})
    except FileNotFoundError:
        logger.warning("config.yaml not found, using default thresholds")
        thresholds = {}
    
    # Default thresholds
    default_thresholds = {
        "roc_auc": 0.85,
        "f1_score": 0.75
    }
    thresholds = {**default_thresholds, **thresholds}
    
    # Call promotion function
    try:
        promotion_result = promote_to_champion(
            model_uri=candidate_info["model_uri"],
            run_id=candidate_info["run_id"],
            metrics=candidate_info["metrics"],
            thresholds=thresholds
        )
        
        result = {
            "promoted": promotion_result["promoted"],
            "model_name": "champion",
            "version": promotion_result.get("version"),
            "run_id": candidate_info["run_id"],
            "reason": promotion_result.get("reason", "unknown"),
            "timestamp": pendulum.now().isoformat(),
            "task": "promote_if_approved"
        }
        
        if promotion_result["promoted"]:
            logger.info(f"Model promoted successfully to version {result['version']}")
        else:
            logger.info(f"Model not promoted: {result['reason']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to promote model: {str(e)}")
        raise


def reload_fastapi(**context) -> Dict[str, Any]:
    """
    Reload FastAPI service if model was promoted.
    
    Returns:
        Dict containing reload status, method used, and timestamp
    """
    import requests
    
    logger.info("Checking if FastAPI reload is needed")
    
    # Pull promotion info from previous task
    promotion_info = context["task_instance"].xcom_pull(task_ids="promote_if_approved")
    
    if not promotion_info or not promotion_info.get("promoted", False):
        logger.info("Model was not promoted, skipping FastAPI reload")
        return {
            "reloaded": False,
            "method": "none",
            "timestamp": pendulum.now().isoformat(),
            "task": "reload_fastapi"
        }
    
    fastapi_reload_url = os.getenv("FASTAPI_RELOAD_URL", "http://fastapi:8000/admin/reload")
    
    # Try HTTP reload first
    try:
        logger.info(f"Attempting HTTP reload at {fastapi_reload_url}")
        response = requests.post(fastapi_reload_url, timeout=30)
        response.raise_for_status()
        
        logger.info("FastAPI reloaded successfully via HTTP")
        return {
            "reloaded": True,
            "method": "http",
            "timestamp": pendulum.now().isoformat(),
            "task": "reload_fastapi"
        }
        
    except Exception as e:
        logger.warning(f"HTTP reload failed: {str(e)}, attempting container restart")
        
        # Fallback to container restart
        try:
            result = subprocess.run(
                ["docker", "compose", "restart", "mlops-fastapi"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd="/opt/airflow"
            )
            
            if result.returncode == 0:
                logger.info("FastAPI restarted successfully via docker compose")
                return {
                    "reloaded": True,
                    "method": "restart",
                    "timestamp": pendulum.now().isoformat(),
                    "task": "reload_fastapi"
                }
            else:
                logger.warning(f"Docker restart failed: {result.stderr}")
                
        except Exception as restart_error:
            logger.warning(f"Container restart failed: {str(restart_error)}")
    
    # If both methods failed, log warning but don't fail the DAG
    logger.warning("All reload methods failed, but continuing DAG execution")
    return {
        "reloaded": False,
        "method": "failed",
        "timestamp": pendulum.now().isoformat(),
        "task": "reload_fastapi"
    }


# Define tasks
wait_for_training_success = ExternalTaskSensor(
    task_id="wait_for_training_success",
    external_dag_id="training_pipeline",
    external_task_id="model_validation",
    # Align with training DAG's daily schedule (runs at midnight)
    execution_date_fn=lambda dt: dt.replace(hour=0, minute=0, second=0, microsecond=0),
    allowed_states=['success'],
    timeout=timedelta(hours=6),
    poke_interval=60,
    mode="reschedule",
    dag=dag,
)

fetch_latest_candidate_task = PythonOperator(
    task_id="fetch_latest_candidate",
    python_callable=fetch_latest_candidate,
    dag=dag,
)

promote_if_approved_task = PythonOperator(
    task_id="promote_if_approved",
    python_callable=promote_if_approved,
    dag=dag,
)

reload_fastapi_task = PythonOperator(
    task_id="reload_fastapi",
    python_callable=reload_fastapi,
    dag=dag,
)

# Set dependencies
wait_for_training_success >> fetch_latest_candidate_task >> promote_if_approved_task >> reload_fastapi_task