"""Drift Detection DAG for monitoring data and concept drift."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to Python path - support both local and Docker environments
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


START_DATE = pendulum.datetime(2025, 8, 29, tz="Asia/Manila")

# Default arguments for the DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": START_DATE,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email": ["your-email@example.com"],
}


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


# Load configuration
config = load_config()
airflow_config = config.get("airflow", {})
drift_config = config.get("drift", {})
monitoring_config = config.get("monitoring", {})

# DAG definition
dag = DAG(
    "drift_detection",
    default_args=default_args,
    description="Monitor data and concept drift using Evidently AI",
    schedule=airflow_config.get("drift_check_schedule", "@hourly"),
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "drift", "evidently"],
)


def check_data_availability(**context):
    """Check if reference and current datasets are available."""
    from pathlib import Path

    reference_path = drift_config.get("reference_data", {}).get(
        "path", "data/reference/reference.parquet"
    )
    current_batch_path = drift_config.get("current_batch", {}).get(
        "output_path", "data/current/current_batch.parquet"
    )

    # Check if reference dataset exists
    if not Path(reference_path).exists():
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

    # Check if current batch exists
    if not Path(current_batch_path).exists():
        raise FileNotFoundError(f"Current batch not found: {current_batch_path}")

    print(f"✓ Reference dataset found: {reference_path}")
    print(f"✓ Current batch found: {current_batch_path}")

    return {"reference_path": reference_path, "current_path": current_batch_path}


def generate_drift_reports(**context):
    """Generate Evidently AI drift reports."""
    from src.monitoring.generate_drift import DriftReportGenerator

    # Get paths from previous task
    ti = context["ti"]
    data_paths = ti.xcom_pull(task_ids="check_data_availability")

    reference_path = data_paths["reference_path"]
    current_path = data_paths["current_path"]
    target_column = config.get("data", {}).get("target_column", "y")

    print("Generating drift reports...")
    print(f"Reference: {reference_path}")
    print(f"Current: {current_path}")
    print(f"Target column: {target_column}")

    # Initialize drift report generator
    generator = DriftReportGenerator()

    # Generate all reports
    results = generator.generate_all_reports(
        reference_path=reference_path,
        current_path=current_path,
        target_column=target_column,
    )

    # Extract key information for downstream tasks
    drift_results = results["drift_results"]
    report_paths = results["report_paths"]

    print("Drift detection completed:")
    print(f"- Data drift detected: {drift_results.get('data_drift_detected', False)}")
    print(
        f"- Target drift detected: {drift_results.get('target_drift_detected', False)}"
    )
    print(f"- Reports generated: {len(report_paths)}")

    # Return results for downstream tasks
    return {
        "drift_results": drift_results,
        "report_paths": report_paths,
        "alerts": drift_results.get("alerts", []),
    }


def analyze_drift_thresholds(**context):
    """Analyze drift results against configured thresholds."""
    ti = context["ti"]
    drift_output = ti.xcom_pull(task_ids="generate_drift_reports")

    drift_results = drift_output["drift_results"]
    alerts = drift_output["alerts"]

    # Get thresholds from configuration
    # thresholds = drift_config.get("thresholds", {})
    # data_drift_threshold = thresholds.get("data_drift_p_value", 0.05)
    # target_drift_threshold = thresholds.get("target_drift_p_value", 0.05)

    # Analyze results
    analysis = {
        "drift_detected": False,
        "alert_level": "INFO",
        "recommendations": [],
        "immediate_action_required": False,
    }

    # Check data drift
    if drift_results.get("data_drift_detected", False):
        analysis["drift_detected"] = True
        analysis["alert_level"] = "WARNING"
        analysis["recommendations"].append(
            "Review data pipeline for distribution changes"
        )
        print("⚠️ Data drift detected!")

    # # Check target drift
    # if drift_results.get("target_drift_detected", False):
    #     analysis["drift_detected"] = True
    #     analysis["alert_level"] = "CRITICAL"
    #     analysis["immediate_action_required"] = True
    #     analysis["recommendations"].append(
    #         "Consider model retraining due to concept drift"
    #     )
    #     print("🚨 Target drift detected - immediate attention required!")

    # Check number of alerts
    if len(alerts) > 3:
        analysis["alert_level"] = "WARNING"
        analysis["recommendations"].append(
            "Multiple drift signals detected - investigate data quality"
        )

    print("Drift analysis completed:")
    print(f"- Alert level: {analysis['alert_level']}")
    print(f"- Immediate action required: {analysis['immediate_action_required']}")
    print(f"- Recommendations: {analysis['recommendations']}")

    return analysis


def send_drift_alerts(**context):
    """Send alerts based on drift analysis results."""
    ti = context["ti"]
    drift_output = ti.xcom_pull(task_ids="generate_drift_reports")
    analysis = ti.xcom_pull(task_ids="analyze_drift_thresholds")

    # Check if alerting is enabled
    monitoring_enabled = monitoring_config.get("drift_detection", {}).get(
        "enabled", True
    )
    if not monitoring_enabled:
        print("Drift monitoring alerts disabled in configuration")
        return

    alert_level = analysis["alert_level"]
    immediate_action = analysis["immediate_action_required"]

    # Prepare alert message
    drift_results = drift_output["drift_results"]
    report_paths = drift_output["report_paths"]

    alert_message = f"""
    Drift Detection Alert - {alert_level}

    Detection Results:
    - Data drift detected: {drift_results.get('data_drift_detected', False)}
    - Target drift detected: {drift_results.get('target_drift_detected', False)}
    - Alert level: {alert_level}
    - Immediate action required: {immediate_action}

    Recommendations:
    {chr(10).join(['- ' + rec for rec in analysis['recommendations']])}

    Generated Reports:
    {chr(10).join([f'- {name}: {path}' for name, path in report_paths.items()])}

    Timestamp: {context['ds']} {context['ts']}
    """

    print("Drift alert prepared:")
    print(alert_message)

    # Log alert to Airflow logs
    if alert_level in ["WARNING", "CRITICAL"]:
        print(f"🚨 {alert_level} ALERT: Drift detected!")
    else:
        print("ℹ️ No significant drift detected")

    return {
        "alert_sent": alert_level in ["WARNING", "CRITICAL"],
        "alert_level": alert_level,
        "message": alert_message,
    }


def cleanup_old_reports(**context):
    """Clean up old drift reports to save disk space."""
    from datetime import timedelta
    from pathlib import Path

    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("Reports directory does not exist")
        return

    # Keep reports for last 30 days
    cutoff_date = datetime.now() - timedelta(days=30)

    cleaned_files = 0
    for report_file in reports_dir.glob("*.html"):
        if report_file.stat().st_mtime < cutoff_date.timestamp():
            report_file.unlink()
            cleaned_files += 1

    print(f"Cleaned up {cleaned_files} old drift reports")
    return {"cleaned_files": cleaned_files}


# Task 1: Check data availability
check_data_task = PythonOperator(
    task_id="check_data_availability",
    python_callable=check_data_availability,
    dag=dag,
    doc_md="""
    ## Check Data Availability

    Verifies that both reference and current datasets are available for drift detection.

    **Inputs:**
    - Reference dataset: `data/reference/reference.parquet`
    - Current batch: `data/current/current_batch.parquet`

    **Outputs:**
    - Paths to validated datasets
    """,
)

# Task 2: Generate drift reports
generate_reports_task = PythonOperator(
    task_id="generate_drift_reports",
    python_callable=generate_drift_reports,
    dag=dag,
    doc_md="""
    ## Generate Drift Reports

    Uses Evidently AI to generate comprehensive drift detection reports.

    **Generated Reports:**
    - Data drift report (feature distribution changes)
    - Target drift report (concept drift)
    - Data quality report (missing values, data types)
    - Drift tests report (automated thresholds)

    **Outputs:**
    - HTML reports saved to `reports/` directory
    - Reports logged as MLflow artifacts
    - Drift metrics and alerts
    """,
)

# Task 3: Analyze drift thresholds
analyze_thresholds_task = PythonOperator(
    task_id="analyze_drift_thresholds",
    python_callable=analyze_drift_thresholds,
    dag=dag,
    doc_md="""
    ## Analyze Drift Thresholds

    Analyzes drift detection results against configured thresholds and determines alert levels.

    **Alert Levels:**
    - INFO: No significant drift detected
    - WARNING: Data drift detected, monitoring recommended
    - CRITICAL: Target drift detected, immediate action required
    """,
)

# Task 4: Send drift alerts
send_alerts_task = PythonOperator(
    task_id="send_drift_alerts",
    python_callable=send_drift_alerts,
    dag=dag,
    doc_md="""
    ## Send Drift Alerts

    Sends alerts based on drift analysis results through configured channels.

    **Alert Channels:**
    - Airflow logs (always)
    - Email notifications (if configured)
    - Slack/Teams (if configured)
    """,
)

# Task 5: Cleanup old reports
cleanup_task = PythonOperator(
    task_id="cleanup_old_reports",
    python_callable=cleanup_old_reports,
    dag=dag,
    doc_md="""
    ## Cleanup Old Reports

    Removes drift reports older than 30 days to manage disk space.
    """,
)


# # Optional: Email notification for critical alerts
# def should_send_email(**context):
#     """Determine if email should be sent based on alert level."""
#     ti = context["ti"]
#     alert_output = ti.xcom_pull(task_ids="send_drift_alerts")
#     return alert_output.get("alert_level") == "CRITICAL"


# email_task = EmailOperator(
#     task_id="send_email_alert",
#     to=airflow_config.get("email", ["gclucos@gmail.com"]),
#     subject="CRITICAL: Drift Detection Alert - Immediate Action Required",
#     html_content="""
#     <h2>Critical Drift Alert</h2>
#     <p>Target drift has been detected in the production model.</p>
#     <p><strong>Immediate action required:</strong> Review model performance and consider retraining.</p>
#     <p>Check the Airflow logs for detailed drift analysis and report locations.</p>
#     <p>Timestamp: {{ ds }} {{ ts }}</p>
#     """,
#     dag=dag,
#     trigger_rule="none_failed",  # Only send if previous tasks succeeded
# )

# Task dependencies
(
    check_data_task
    >> generate_reports_task
    >> analyze_thresholds_task
    >> send_alerts_task
    >> cleanup_task
)


# send_alerts_task >> email_task
