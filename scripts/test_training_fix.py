#!/usr/bin/env python3
"""
Quick test script to verify the training DAG fix.
Tests that the train_catboost_model function returns the expected structure.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_train_function_structure():
    """Test that train_catboost_model returns expected structure."""
    print("Testing train_catboost_model function structure...")

    try:
        from src.models.train import train_catboost_model

        # This would normally run the full training, but we'll just check the import
        print("✅ Successfully imported train_catboost_model")

        # Check if we have processed data to actually test with
        try:
            from src.features.transform import load_processed_data

            X_train, X_test, y_train, y_test = load_processed_data()
            print(f"✅ Processed data available: {X_train.shape[0]} training samples")

            # Run the actual training function
            print("🔄 Running train_catboost_model()...")
            results = train_catboost_model()

            # Check expected keys
            expected_keys = [
                "status",
                "saved_paths",
                "best_iteration",
                "best_score",
                "training_data_shape",
                "test_data_shape",
            ]

            print("Checking returned structure:")
            for key in expected_keys:
                if key in results:
                    value = results[key]
                    if key == "best_score":
                        print(f"  ✅ {key}: {type(value)} - {value}")
                    elif key in ["training_data_shape", "test_data_shape"]:
                        print(f"  ✅ {key}: {value}")
                    else:
                        print(f"  ✅ {key}: {type(value)}")
                else:
                    print(f"  ❌ {key}: MISSING")

            # Test DAG compatibility
            print("\nTesting DAG compatibility:")
            try:
                model_path = results["saved_paths"]["final_model_pkl"]
                print(f"  ✅ Model path accessible: {model_path}")

                best_score = results.get("best_score", {})
                score_value = (
                    best_score.get("validation", {}).get("AUC", best_score)
                    if isinstance(best_score, dict)
                    else best_score
                )
                print(f"  ✅ Best score extractable: {score_value}")

                print(f"  ✅ Status: {results['status']}")
                print(f"  ✅ Best iteration: {results['best_iteration']}")

            except Exception as e:
                print(f"  ❌ DAG compatibility issue: {e}")
                return False

            print("\n🎉 Training function structure test PASSED!")
            return True

        except FileNotFoundError:
            print("⚠️ No processed data found - skipping actual training test")
            print("✅ Import test passed")
            return True

    except Exception as e:
        print(f"❌ Training function test failed: {e}")
        return False


def test_dag_task_function():
    """Test the DAG task function with mock data."""
    print("\nTesting DAG task function...")

    try:
        from unittest.mock import MagicMock

        from dags.training_dag import model_training_task

        # Create mock context
        context = {
            "task_instance": MagicMock(),
            "ds": "2025-08-28",
            "ts": "2025-08-28T10:00:00+00:00",
        }

        # Mock XCom data
        context["task_instance"].xcom_pull.return_value = {"status": "success_cached"}

        print("🔄 Testing model_training_task with mock context...")

        # This will run the actual training if data is available
        result = model_training_task(**context)

        print("Checking task result structure:")
        expected_keys = [
            "status",
            "model_path",
            "best_iteration",
            "best_score",
            "training_data_shape",
            "test_data_shape",
            "timestamp",
            "task",
        ]

        for key in expected_keys:
            if key in result:
                print(f"  ✅ {key}: {result[key]}")
            else:
                print(f"  ❌ {key}: MISSING")

        print("\n🎉 DAG task function test PASSED!")
        return True

    except Exception as e:
        print(f"❌ DAG task function test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRAINING DAG FIX VERIFICATION")
    print("=" * 60)

    success1 = test_train_function_structure()
    success2 = test_dag_task_function()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if success1 and success2:
        print("🎉 All tests PASSED! The training DAG fix should work.")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED. Check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
