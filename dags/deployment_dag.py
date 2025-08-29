"""
MLOps Deployment DAG for Bank Term Deposit Prediction (push-based)

This DAG is triggered by training_pipeline (TriggerDagRunOperator) after
`model_validation` succeeds. It:
- Verifies upstream context from dag_run.conf
- Fetches the latest candidate model from MLflow
- Evaluates against thresholds and promotes to MLflow Model Registry
- Reloads FastAPI (HTTP endpoint or docker compose restart fallback)
"""

import logging
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

# Add project root to Python path for Airflow
project_root = Path(__file__).parent.parent
src_path = os.path.join(project_root, "src")
airflow_src_path = "/opt/airflow/src"

# Add both local and Docker container paths
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if airflow_src_path not in sys.path:
    sys.path.insert(0, airflow_src_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_DATE = pendulum.datetime(2025, 8, 29, tz="Asia/Manila")

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
    description="MLOps deployment pipeline for model promotion and API reload (push-based)",
    schedule=None,  # push-triggered; no schedule
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    tags=["ml", "deployment", "mlflow", "registry"],
    doc_md=__doc__,
)


# ----------------------
# Helpers / Task funcs
# ----------------------
def verify_upstream(**context) -> Dict[str, Any]:
    """
    Verifies the upstream trigger context from training_pipeline.
    Expects dag_run.conf to include:
      - validation_status == "success"
      - training_dag_id == "training_pipeline" (optional check)
      - training_run_id, training_logical_date (optional)
    If validation failed or missing, skip the DAG.
    """
    conf = context.get("dag_run").conf or {}
    status = conf.get("validation_status")
    src_dag = conf.get("training_dag_id")
    if status != "success":
        logger.warning(f"Deployment skipped: validation_status={status}")
        raise AirflowSkipException("Upstream validation not successful; skipping.")
    if src_dag and src_dag != "training_pipeline":
        logger.warning(f"Deployment triggered by unexpected DAG: {src_dag}")
    logger.info(f"Verified upstream trigger: {conf}")
    return conf


def fetch_latest_candidate(**context) -> Dict[str, Any]:
    """
    Fetch the latest candidate model from MLflow for promotion evaluation.
    Returns dict with run_id, model_uri, metrics.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    logger.info("Fetching latest candidate model from MLflow")
    client = MlflowClient()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-marketing-catboost")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else "0"
    if not experiment:
        logger.warning(
            f"Experiment '{experiment_name}' not found, using default experiment_id=0"
        )

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["end_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No completed runs found in MLflow; cannot deploy.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
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
        "task": "fetch_latest_candidate",
    }
    logger.info(
        f"Candidate run_id={run_id} ROC-AUC={metrics['roc_auc']:.4f} F1={metrics['f1_score']:.4f}"
    )
    return result


def promote_if_approved(**context) -> Dict[str, Any]:
    """
    Evaluate model against thresholds and promote to MLflow Model Registry if approved.
    """
    import yaml

    from src.deployment.promote import promote_to_champion

    logger.info("Evaluating model for promotion")
    ti = context["task_instance"]
    candidate = ti.xcom_pull(task_ids="fetch_latest_candidate")
    if not candidate:
        raise ValueError("fetch_latest_candidate returned no data")

    # Load thresholds from config (optional)
    thresholds = {}
    try:
        cfg_path = os.getenv("PIPELINE_CONFIG", "/opt/airflow/config.yaml")
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        thresholds = (config or {}).get("model", {}).get(
            "promotion_thresholds", {}
        ) or {}
    except FileNotFoundError:
        logger.warning("config.yaml not found; using default thresholds")

    default_thresholds = {"roc_auc": 0.85, "f1_score": 0.75}
    thresholds = {**default_thresholds, **thresholds}

    promotion_result = promote_to_champion(
        model_uri=candidate["model_uri"],
        run_id=candidate["run_id"],
        metrics=candidate["metrics"],
        thresholds=thresholds,
    )

    result = {
        "promoted": promotion_result.get("promoted", False),
        "model_name": promotion_result.get("model_name", "champion"),
        "version": promotion_result.get("version"),
        "run_id": candidate["run_id"],
        "reason": promotion_result.get("reason", "unknown"),
        "timestamp": pendulum.now().isoformat(),
        "task": "promote_if_approved",
    }

    if result["promoted"]:
        logger.info(f"Model promoted to {result['model_name']} v{result['version']}")
    else:
        logger.info(f"Model NOT promoted: {result['reason']}")
    return result


def reload_fastapi(**context) -> Dict[str, Any]:
    """
    Reload FastAPI service if model was promoted. Tries HTTP admin endpoint first,
    falls back to `docker compose restart mlops-fastapi`.
    """
    import requests

    ti = context["task_instance"]
    promo = ti.xcom_pull(task_ids="promote_if_approved")
    if not promo or not promo.get("promoted", False):
        logger.info("No promotion occurred; skipping FastAPI reload.")
        return {
            "reloaded": False,
            "method": "none",
            "task": "reload_fastapi",
            "timestamp": pendulum.now().isoformat(),
        }

    fastapi_reload_url = os.getenv(
        "FASTAPI_RELOAD_URL", "http://fastapi:8000/admin/reload"
    )

    # Attempt HTTP reload
    try:
        logger.info(f"Reloading FastAPI via HTTP: {fastapi_reload_url}")
        resp = requests.post(fastapi_reload_url, timeout=30)
        resp.raise_for_status()
        logger.info("FastAPI reloaded via HTTP")
        return {
            "reloaded": True,
            "method": "http",
            "task": "reload_fastapi",
            "timestamp": pendulum.now().isoformat(),
        }
    except Exception as http_err:
        logger.warning(f"HTTP reload failed: {http_err}; trying docker restart.")

    # Fallback: docker compose restart
    try:
        result = subprocess.run(
            ["docker", "compose", "restart", "mlops-fastapi"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/opt/airflow",
        )
        if result.returncode == 0:
            logger.info("FastAPI restarted via docker compose")
            return {
                "reloaded": True,
                "method": "restart",
                "task": "reload_fastapi",
                "timestamp": pendulum.now().isoformat(),
            }
        logger.warning(f"Docker restart failed: {result.stderr}")
    except Exception as restart_err:
        logger.warning(f"Container restart failed: {restart_err}")

    logger.warning("All reload methods failed; continuing DAG.")
    return {
        "reloaded": False,
        "method": "failed",
        "task": "reload_fastapi",
        "timestamp": pendulum.now().isoformat(),
    }


# ----------------------
# Tasks
# ----------------------
verify_upstream_task = PythonOperator(
    task_id="verify_upstream",
    python_callable=verify_upstream,
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

# Flow
(
    verify_upstream_task
    >> fetch_latest_candidate_task
    >> promote_if_approved_task
    >> reload_fastapi_task
)
