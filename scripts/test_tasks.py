#!/usr/bin/env python3
"""
Task Testing Utilities for Bank Term Deposit Prediction MLOps Pipeline

This script allows testing individual Airflow tasks outside of the Airflow environment
with mock contexts and data, useful for development and debugging.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockTaskInstance:
    """Mock Airflow TaskInstance for testing."""

    def __init__(self):
        self.xcom_data = {}

    def xcom_pull(self, task_ids: str, key: str = None) -> Any:
        """Mock XCom pull operation."""
        if task_ids in self.xcom_data:
            data = self.xcom_data[task_ids]
            if key:
                return data.get(key) if isinstance(data, dict) else None
            return data
        return None

    def xcom_push(self, key: str, value: Any) -> None:
        """Mock XCom push operation."""
        if not hasattr(self, "task_id"):
            self.task_id = "test_task"
        if self.task_id not in self.xcom_data:
            self.xcom_data[self.task_id] = {}
        self.xcom_data[self.task_id][key] = value

    def set_xcom_data(self, task_id: str, data: Any) -> None:
        """Set XCom data for testing."""
        self.xcom_data[task_id] = data


class TaskTester:
    """Utility class for testing Airflow tasks."""

    def __init__(self):
        self.mock_ti = MockTaskInstance()
        self.context = self._create_mock_context()

    def _create_mock_context(self) -> Dict[str, Any]:
        """Create a mock Airflow context."""
        execution_date = datetime.now() - timedelta(days=1)

        context = {
            "task_instance": self.mock_ti,
            "ds": execution_date.strftime("%Y-%m-%d"),
            "ts": execution_date.isoformat(),
            "execution_date": execution_date,
            "dag_run": MagicMock(),
            "task": MagicMock(),
            "params": {},
            "var": {"value": lambda x, default=None: default},
        }

        # Add mock methods
        context["task"].task_id = "test_task"
        context["dag_run"].dag_id = "test_dag"

        return context

    def set_upstream_data(self, task_id: str, data: Any) -> None:
        """Set mock data from upstream tasks."""
        self.mock_ti.set_xcom_data(task_id, data)

    def test_data_ingestion(self) -> Dict[str, Any]:
        """Test data ingestion task."""
        logger.info("Testing data ingestion task...")

        try:
            from dags.training_dag import data_ingestion_task

            result = data_ingestion_task(**self.context)
            logger.info(f"Data ingestion result: {result}")
            return result
        except Exception as e:
            logger.error(f"Data ingestion test failed: {e}")
            raise

    def test_data_transformation(
        self, ingestion_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Test data transformation task."""
        logger.info("Testing data transformation task...")

        # Set mock upstream data
        if ingestion_data is None:
            ingestion_data = {
                "status": "success",
                "dataset_shape": (1000, 20),
                "dataset_columns": ["age", "job", "marital", "education", "y"],
                "raw_data_path": "data/raw/train.csv",
            }

        self.set_upstream_data("data_ingestion", ingestion_data)

        try:
            from dags.training_dag import data_transformation_task

            result = data_transformation_task(**self.context)
            logger.info(f"Data transformation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Data transformation test failed: {e}")
            raise

    def test_model_training(
        self, transformation_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Test model training task."""
        logger.info("Testing model training task...")

        # Set mock upstream data
        if transformation_data is None:
            transformation_data = {
                "status": "success_cached",
                "X_train_shape": (800, 19),
                "X_test_shape": (200, 19),
                "y_train_shape": (800,),
                "y_test_shape": (200,),
                "processed_data_path": "data/processed/",
            }

        self.set_upstream_data("data_transformation", transformation_data)

        try:
            from dags.training_dag import model_training_task

            result = model_training_task(**self.context)
            logger.info(f"Model training result: {result}")
            return result
        except Exception as e:
            logger.error(f"Model training test failed: {e}")
            raise

    def test_model_validation(
        self, training_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Test model validation task."""
        logger.info("Testing model validation task...")

        # Set mock upstream data
        if training_data is None:
            training_data = {
                "status": "success",
                "model_path": "models/catboost_model.pkl",
                "best_iteration": 100,
                "best_score": 0.85,
            }

        self.set_upstream_data("model_training", training_data)

        try:
            from dags.training_dag import model_validation_task

            result = model_validation_task(**self.context)
            logger.info(f"Model validation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Model validation test failed: {e}")
            raise

    def test_full_pipeline(self) -> Dict[str, Any]:
        """Test the full training pipeline."""
        logger.info("Testing full training pipeline...")

        results = {}

        try:
            # Test data ingestion
            logger.info("Step 1: Data Ingestion")
            ingestion_result = self.test_data_ingestion()
            results["data_ingestion"] = ingestion_result

            # Test data transformation
            logger.info("Step 2: Data Transformation")
            transformation_result = self.test_data_transformation(ingestion_result)
            results["data_transformation"] = transformation_result

            # Test model training
            logger.info("Step 3: Model Training")
            training_result = self.test_model_training(transformation_result)
            results["model_training"] = training_result

            # Test model validation
            logger.info("Step 4: Model Validation")
            validation_result = self.test_model_validation(training_result)
            results["model_validation"] = validation_result

            logger.info("✅ Full pipeline test completed successfully!")
            return results

        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
            results["error"] = str(e)
            return results

    def test_drift_detection(self) -> Dict[str, Any]:
        """Test drift detection tasks."""
        logger.info("Testing drift detection task...")

        try:
            from dags.drift_dag import check_data_availability, generate_drift_reports

            # Test data availability check
            logger.info("Checking data availability...")
            data_result = check_data_availability(**self.context)
            logger.info(f"Data availability result: {data_result}")

            # Mock data paths for drift report generation
            self.set_upstream_data("check_data_availability", data_result)

            # Test drift report generation (may fail if data files don't exist)
            logger.info("Generating drift reports...")
            try:
                drift_result = generate_drift_reports(**self.context)
                logger.info(f"Drift generation result: {drift_result}")
                return {"data_check": data_result, "drift_reports": drift_result}
            except FileNotFoundError as e:
                logger.warning(
                    f"Drift report generation skipped (missing data files): {e}"
                )
                return {"data_check": data_result, "drift_reports": None}

        except ImportError as e:
            logger.error(f"Drift DAG not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Drift detection test failed: {e}")
            raise

    def save_results(self, results: Dict[str, Any], filename: str = None) -> None:
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        output_path = project_root / "reports" / filename
        output_path.parent.mkdir(exist_ok=True)

        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=json_serializer)

        logger.info(f"Test results saved to: {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Airflow tasks outside of Airflow environment"
    )
    parser.add_argument(
        "task",
        choices=[
            "ingestion",
            "transformation",
            "training",
            "validation",
            "pipeline",
            "drift",
        ],
        help="Task to test",
    )
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    parser.add_argument("--output", help="Output filename for results")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tester = TaskTester()

    try:
        if args.task == "ingestion":
            results = {"data_ingestion": tester.test_data_ingestion()}
        elif args.task == "transformation":
            results = {"data_transformation": tester.test_data_transformation()}
        elif args.task == "training":
            results = {"model_training": tester.test_model_training()}
        elif args.task == "validation":
            results = {"model_validation": tester.test_model_validation()}
        elif args.task == "pipeline":
            results = tester.test_full_pipeline()
        elif args.task == "drift":
            results = {"drift_detection": tester.test_drift_detection()}

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2, default=str))

        if args.save:
            tester.save_results(results, args.output)

    except Exception as e:
        logger.error(f"Task test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
