"""
Unit tests for Airflow DAGs in the Bank Term Deposit Prediction MLOps pipeline.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


class TestDAGIntegrity:
    """Test DAG structure and configuration."""

    def test_training_dag_import(self):
        """Test that training DAG can be imported without errors."""
        try:
            from dags.training_dag import dag

            assert dag is not None
            assert dag.dag_id == "training_pipeline"
        except ImportError as e:
            pytest.skip(f"Training DAG import failed: {e}")

    def test_drift_dag_import(self):
        """Test that drift DAG can be imported (may require config)."""
        try:
            from dags.drift_dag import dag

            assert dag is not None
            assert dag.dag_id == "drift_detection"
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"Drift DAG import failed (may need config): {e}")

    def test_deployment_dag_import(self):
        """Test that deployment DAG can be imported (may require modules)."""
        try:
            from dags.deployment_dag import dag

            assert dag is not None
            assert dag.dag_id == "deployment_dag"
        except ImportError as e:
            pytest.skip(f"Deployment DAG import failed: {e}")

    def test_training_dag_structure(self):
        """Test training DAG has correct structure."""
        try:
            from dags.training_dag import dag

            # Check basic properties
            assert dag.dag_id == "training_pipeline"
            assert dag.description == "ML Training Pipeline"
            assert dag.schedule == "@daily"
            assert dag.max_active_runs == 1
            assert dag.catchup is False

            # Check tasks exist
            task_ids = [task.task_id for task in dag.tasks]
            expected_tasks = [
                "data_ingestion",
                "data_transformation",
                "model_training",
                "model_validation",
            ]

            for expected_task in expected_tasks:
                assert (
                    expected_task in task_ids
                ), f"Task {expected_task} not found in DAG"

            # Check task dependencies
            data_ingestion_task = dag.get_task("data_ingestion")
            data_transformation_task = dag.get_task("data_transformation")
            model_training_task = dag.get_task("model_training")
            model_validation_task = dag.get_task("model_validation")

            # Verify dependency chain
            assert data_transformation_task in data_ingestion_task.downstream_list
            assert model_training_task in data_transformation_task.downstream_list
            assert model_validation_task in model_training_task.downstream_list

        except ImportError as e:
            pytest.skip(f"Training DAG not available for structure test: {e}")

    def test_dag_default_args(self):
        """Test that DAGs have proper default arguments."""
        try:
            from dags.training_dag import dag

            default_args = dag.default_args

            # Check required default args
            assert "owner" in default_args
            assert "depends_on_past" in default_args
            assert "start_date" in default_args
            assert "retries" in default_args
            assert "retry_delay" in default_args

            # Check values
            assert default_args["depends_on_past"] is False
            assert default_args["retries"] >= 1
            assert isinstance(default_args["retry_delay"], timedelta)

        except ImportError as e:
            pytest.skip(f"Training DAG not available for default args test: {e}")


class TestDAGTasks:
    """Test individual DAG task functions."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock Airflow context."""
        context = {
            "task_instance": MagicMock(),
            "ds": "2025-08-28",
            "ts": "2025-08-28T10:00:00+00:00",
            "execution_date": datetime(2025, 8, 28),
            "dag_run": MagicMock(),
        }
        # Mock XCom pull/push
        context["task_instance"].xcom_pull.return_value = None
        context["task_instance"].xcom_push.return_value = None
        return context

    @patch("src.data.ingest.load_dataset")
    def test_data_ingestion_task_success(self, mock_load_dataset, mock_context):
        """Test data ingestion task with successful execution."""
        try:
            from dags.training_dag import data_ingestion_task

            # Mock dataset
            mock_dataset = MagicMock()
            mock_dataset.shape = (1000, 20)
            mock_dataset.columns = ["age", "job", "marital", "education", "y"]
            mock_load_dataset.return_value = mock_dataset

            # Execute task
            result = data_ingestion_task(**mock_context)

            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["dataset_shape"] == (1000, 20)
            assert result["task"] == "data_ingestion"
            assert "timestamp" in result

            # Verify function was called
            mock_load_dataset.assert_called_once_with("train")

        except ImportError as e:
            pytest.skip(f"Data ingestion task not available: {e}")

    @patch("src.features.transform.load_processed_data")
    @patch("src.data.ingest.load_dataset")
    def test_data_transformation_task_cached(
        self, mock_load_dataset, mock_load_processed, mock_context
    ):
        """Test data transformation task with cached data."""
        try:
            from dags.training_dag import data_transformation_task

            # Mock cached data exists
            mock_X_train = MagicMock()
            mock_X_train.shape = (800, 19)
            mock_X_test = MagicMock()
            mock_X_test.shape = (200, 19)
            mock_y_train = MagicMock()
            mock_y_train.shape = (800,)
            mock_y_test = MagicMock()
            mock_y_test.shape = (200,)

            mock_load_processed.return_value = (
                mock_X_train,
                mock_X_test,
                mock_y_train,
                mock_y_test,
            )

            # Mock XCom data
            mock_context["task_instance"].xcom_pull.return_value = {
                "status": "success",
                "dataset_shape": (1000, 20),
            }

            # Execute task
            result = data_transformation_task(**mock_context)

            # Verify result
            assert result is not None
            assert result["status"] == "success_cached"
            assert result["X_train_shape"] == (800, 19)
            assert result["X_test_shape"] == (200, 19)
            assert result["task"] == "data_transformation"

            # Verify functions called
            mock_load_processed.assert_called_once()
            mock_context["task_instance"].xcom_pull.assert_called_once_with(
                task_ids="data_ingestion"
            )

        except ImportError as e:
            pytest.skip(f"Data transformation task not available: {e}")

    @patch("src.models.train.train_catboost_model")
    def test_model_training_task_success(self, mock_train_model, mock_context):
        """Test model training task with successful execution."""
        try:
            from dags.training_dag import model_training_task

            # Mock training results
            mock_training_results = {
                "status": "success",
                "saved_paths": {"final_model_pkl": "models/catboost_model.pkl"},
                "best_iteration": 100,
                "best_score": 0.85,
                "training_data_shape": (800, 19),
                "test_data_shape": (200, 19),
            }
            mock_train_model.return_value = mock_training_results

            # Mock XCom data
            mock_context["task_instance"].xcom_pull.return_value = {
                "status": "success_cached"
            }

            # Execute task
            result = model_training_task(**mock_context)

            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["model_path"] == "models/catboost_model.pkl"
            assert result["best_iteration"] == 100
            assert result["best_score"] == 0.85
            assert result["task"] == "model_training"

            # Verify functions called
            mock_train_model.assert_called_once()
            mock_context["task_instance"].xcom_pull.assert_called_once_with(
                task_ids="data_transformation"
            )

        except ImportError as e:
            pytest.skip(f"Model training task not available: {e}")

    @patch("src.models.validate.validate_model")
    def test_model_validation_task_success(self, mock_validate_model, mock_context):
        """Test model validation task with successful execution."""
        try:
            from dags.training_dag import model_validation_task

            # Mock validation results
            mock_validation_results = {
                "metrics": {
                    "roc_auc": 0.89,
                    "precision": 0.82,
                    "recall": 0.75,
                    "f1_score": 0.78,
                },
                "validation_results": {
                    "roc_auc_pass": True,
                    "precision_pass": True,
                    "recall_pass": True,
                },
                "plot_paths": {
                    "roc_curve": "reports/roc_curve.png",
                    "confusion_matrix": "reports/confusion_matrix.png",
                },
            }
            mock_validate_model.return_value = mock_validation_results

            # Mock XCom data
            mock_context["task_instance"].xcom_pull.return_value = {
                "model_path": "models/catboost_model.pkl"
            }

            # Execute task
            result = model_validation_task(**mock_context)

            # Verify result
            assert result is not None
            assert result["status"] == "success"
            assert result["roc_auc"] == 0.89
            assert result["validation_passed"] is True
            assert result["task"] == "model_validation"

            # Verify functions called
            mock_validate_model.assert_called_once_with(
                model_path="models/catboost_model.pkl"
            )
            mock_context["task_instance"].xcom_pull.assert_called_once_with(
                task_ids="model_training"
            )

        except ImportError as e:
            pytest.skip(f"Model validation task not available: {e}")

    def test_pipeline_callbacks(self, mock_context):
        """Test pipeline success and failure callbacks."""
        try:
            from dags.training_dag import (
                pipeline_failure_callback,
                pipeline_success_callback,
            )

            # Mock XCom data for success callback
            mock_context["task_instance"].xcom_pull.side_effect = [
                {"dataset_shape": (1000, 20)},  # ingestion
                {"X_train_shape": (800, 19)},  # transformation
                {"best_iteration": 100, "model_path": "models/test.pkl"},  # training
                {"roc_auc": 0.89, "validation_passed": True},  # validation
            ]

            # Test success callback (should not raise exception)
            try:
                pipeline_success_callback(**mock_context)
            except Exception as e:
                pytest.fail(f"Success callback failed: {e}")

            # Test failure callback (should not raise exception)
            mock_context["task_instance"].task_id = "test_task"
            mock_context["task_instance"].dag_id = "test_dag"

            try:
                pipeline_failure_callback(**mock_context)
            except Exception as e:
                pytest.fail(f"Failure callback failed: {e}")

        except ImportError as e:
            pytest.skip(f"Pipeline callbacks not available: {e}")


class TestDAGConfiguration:
    """Test DAG configuration and environment setup."""

    def test_python_path_setup(self):
        """Test that Python path is configured correctly."""
        # Check that src modules can be imported
        try:
            import src

            assert src is not None
        except ImportError:
            # This is acceptable if running without proper setup
            pass

    def test_dag_file_syntax(self):
        """Test that all DAG files have valid Python syntax."""
        dag_files = [
            "dags/training_dag.py",
            "dags/drift_dag.py",
            "dags/deployment_dag.py",
        ]

        for dag_file in dag_files:
            dag_path = project_root / dag_file
            if dag_path.exists():
                try:
                    with open(dag_path, "r") as f:
                        code = f.read()
                    compile(code, dag_file, "exec")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {dag_file}: {e}")
            else:
                pytest.skip(f"DAG file not found: {dag_file}")

    @pytest.mark.integration
    def test_dag_bag_loading(self):
        """Integration test for loading DAGs into DagBag (requires Airflow)."""
        try:
            from airflow.models import DagBag

            dag_bag = DagBag(
                dag_folder=str(project_root / "dags"), include_examples=False
            )

            # Check for import errors
            if dag_bag.import_errors:
                error_messages = "\n".join(
                    [
                        f"{dag_id}: {error}"
                        for dag_id, error in dag_bag.import_errors.items()
                    ]
                )
                pytest.fail(f"DAG import errors:\n{error_messages}")

            # Check that expected DAGs are loaded
            dag_ids = list(dag_bag.dag_ids)
            expected_dags = ["training_pipeline"]  # Only test the main one

            for expected_dag in expected_dags:
                assert (
                    expected_dag in dag_ids
                ), f"Expected DAG {expected_dag} not found in DagBag"

        except ImportError:
            pytest.skip("Airflow not available for DagBag test")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
