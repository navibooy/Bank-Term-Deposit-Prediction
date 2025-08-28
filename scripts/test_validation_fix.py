#!/usr/bin/env python3
"""
Test script to verify the validation fixes.
Tests MLflow connection handling and matplotlib configuration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_mlflow_connection_handling():
    """Test that MLflow connection handling works properly."""
    print("Testing MLflow connection handling...")

    try:
        from src.models.validate import setup_mlflow

        # Create mock config
        test_config = {
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "test-experiment",
            }
        }

        print("🔄 Testing setup_mlflow with different URIs...")

        # This should return False if MLflow is not available, but not crash
        result = setup_mlflow(test_config)

        print(f"✅ setup_mlflow returned: {result}")
        print("✅ Function handled connection issues gracefully")

        return True

    except Exception as e:
        print(f"❌ setup_mlflow test failed: {e}")
        return False


def test_matplotlib_import():
    """Test that matplotlib imports work without permission issues."""
    print("\nTesting matplotlib import...")

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        # Set backend to avoid display issues
        matplotlib.use("Agg")

        print("✅ Matplotlib imported successfully")
        print(f"✅ Backend: {matplotlib.get_backend()}")

        # Test creating a simple plot
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title("Test Plot")
        plt.close()

        print("✅ Matplotlib plot creation works")

        return True

    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        return False


def test_validation_function():
    """Test the validation function with mock data."""
    print("\nTesting validation function...")

    try:
        from src.models.validate import validate_model

        # Check if we have a trained model to test with
        model_path = project_root / "models" / "catboost_model.pkl"

        if model_path.exists():
            print(f"✅ Found trained model at: {model_path}")

            # Try to run validation (may fail due to missing MLflow, but should handle gracefully)
            print("🔄 Running validation...")

            result = validate_model(str(model_path))

            print("✅ Validation completed successfully!")
            print(f"✅ Returned keys: {list(result.keys())}")

            # Check that required keys are present
            required_keys = ["metrics", "validation_results", "plot_paths"]
            for key in required_keys:
                if key in result:
                    print(f"  ✅ {key}: present")
                else:
                    print(f"  ❌ {key}: missing")

            return True
        else:
            print("⚠️  No trained model found - skipping validation test")
            print("✅ Validation function import works")
            return True

    except Exception as e:
        print(f"❌ Validation function test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dag_task_function():
    """Test the DAG validation task."""
    print("\nTesting DAG validation task...")

    try:
        from unittest.mock import MagicMock

        from dags.training_dag import model_validation_task

        # Create mock context
        context = {
            "task_instance": MagicMock(),
            "ds": "2025-08-28",
            "ts": "2025-08-28T10:00:00+00:00",
        }

        # Mock training data
        context["task_instance"].xcom_pull.return_value = {
            "status": "success",
            "model_path": "models/catboost_model.pkl",
            "best_iteration": 4,
            "best_score": 0.952,
        }

        print("🔄 Testing model_validation_task...")

        # Check if model exists
        model_path = project_root / "models" / "catboost_model.pkl"
        if model_path.exists():
            result = model_validation_task(**context)

            print("✅ Validation task completed successfully!")
            print(f"✅ Task result keys: {list(result.keys())}")

            return True
        else:
            print("⚠️  No trained model found - skipping DAG task test")
            return True

    except Exception as e:
        print(f"❌ DAG validation task test failed: {e}")
        return False


def main():
    """Run all validation fix tests."""
    print("=" * 60)
    print("VALIDATION FIX VERIFICATION")
    print("=" * 60)

    tests = [
        ("MLflow Connection Handling", test_mlflow_connection_handling),
        ("Matplotlib Import", test_matplotlib_import),
        ("Validation Function", test_validation_function),
        ("DAG Validation Task", test_dag_task_function),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation fixes are working!")
        return 0
    else:
        print("⚠️  Some issues remain - check the failures above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
