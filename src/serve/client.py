"""Client code for testing Bank Marketing Prediction API endpoints."""

import logging

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000"


def test_root_endpoint():
    """Test the root endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/")
        logger.info(f"GET / -> Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as error:
        logger.error(f"Error testing root endpoint: {str(error)}")
        return False


def test_health_check():
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        logger.info(f"GET /health -> Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as error:
        logger.error(f"Error testing health endpoint: {str(error)}")
        return False


def test_model_info():
    """Test the model information endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/model")
        logger.info(f"GET /model -> Status: {response.status_code}")

        if response.status_code == 200:
            model_info = response.json()
            logger.info("Model Information:")
            logger.info(f"  Model Name: {model_info.get('model_name')}")
            logger.info(f"  Model Version: {model_info.get('model_version')}")
            logger.info(f"  Hyperparameters: {model_info.get('hyperparameters')}")
            logger.info(f"  Top Features: {model_info.get('top_features')}")
            logger.info(f"  Input Schema: {model_info.get('input_schema')}")
        else:
            logger.warning(f"Model info unavailable: {response.json()}")

        return response.status_code in [
            200,
            503,
        ]  # 503 is acceptable if model not loaded
    except Exception as error:
        logger.error(f"Error testing model info endpoint: {str(error)}")
        return False


def test_prediction_valid_input():
    """Test prediction endpoint with valid input."""
    try:
        prediction_data = {
            "age": 42,
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 1500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "duration": 300,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=prediction_data,
            headers={"Content-Type": "application/json"},
        )

        logger.info(f"POST /predict -> Status: {response.status_code}")

        if response.status_code == 200:
            prediction_result = response.json()
            logger.info("Prediction Result:")
            logger.info(
                f"  Prediction: {prediction_result.get('prediction')} (0=No subscription, 1=Will subscribe)"
            )
            logger.info(f"  Probability: {prediction_result.get('probability'):.4f}")
            logger.info(f"  Model Version: {prediction_result.get('model_version')}")
            logger.info(f"  Timestamp: {prediction_result.get('prediction_timestamp')}")
        else:
            logger.warning(f"Prediction failed: {response.json()}")

        return response.status_code in [200, 503]  # 503 acceptable if model not loaded
    except Exception as error:
        logger.error(f"Error testing prediction with valid input: {str(error)}")
        return False


def test_prediction_invalid_input():
    """Test prediction endpoint with invalid input."""
    try:
        invalid_data = {
            "age": -5,  # Invalid age
            "job": "invalid_job",  # Invalid job type
            "marital": "complicated",  # Invalid marital status
            "education": "phd_in_life",  # Invalid education
            "default": "maybe",  # Invalid default value
            "balance": "lots",  # Invalid balance type
            "housing": "yes",
            "loan": "no",
            "contact": "telepathy",  # Invalid contact type
            "day": 35,  # Invalid day
            "month": "mayan_calendar",  # Invalid month
            "duration": -100,  # Invalid duration
            "campaign": 0,  # Invalid campaign
            "pdays": -2,  # Invalid pdays
            "previous": -1,  # Invalid previous
            "poutcome": "maybe_success",  # Invalid poutcome
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
        )

        logger.info(f"POST /predict (invalid) -> Status: {response.status_code}")

        if response.status_code == 422:  # Validation error
            logger.info("Validation error correctly caught:")
            logger.info(f"Response: {response.json()}")
        else:
            logger.warning(f"Unexpected response for invalid input: {response.json()}")

        return response.status_code in [400, 422]  # Expected validation errors
    except Exception as error:
        logger.error(f"Error testing prediction with invalid input: {str(error)}")
        return False


def test_prediction_missing_fields():
    """Test prediction endpoint with missing required fields."""
    try:
        incomplete_data = {
            "age": 30,
            "job": "admin.",
            # Missing many required fields like marital, education, etc.
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=incomplete_data,
            headers={"Content-Type": "application/json"},
        )

        logger.info(f"POST /predict (missing fields) -> Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")

        return response.status_code in [400, 422]  # Expected validation errors
    except Exception as error:
        logger.error(f"Error testing prediction with missing fields: {str(error)}")
        return False


def test_api_documentation():
    """Test if API documentation is accessible."""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        logger.info(f"GET /docs -> Status: {response.status_code}")

        if response.status_code == 200:
            logger.info("API documentation is accessible at /docs")

        # Test OpenAPI JSON schema
        response = requests.get(f"{BASE_URL}/openapi.json")
        logger.info(f"GET /openapi.json -> Status: {response.status_code}")

        return response.status_code == 200
    except Exception as error:
        logger.error(f"Error testing API documentation: {str(error)}")
        return False


def run_comprehensive_tests():
    """Run all API tests and report results."""
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health_check),
        ("Model Information", test_model_info),
        ("Valid Prediction", test_prediction_valid_input),
        ("Invalid Input Validation", test_prediction_invalid_input),
        ("Missing Fields Validation", test_prediction_missing_fields),
        ("API Documentation", test_api_documentation),
    ]

    results = {}
    logger.info("=== Starting Bank Marketing Prediction API Comprehensive Tests ===")

    for test_name, test_function in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        try:
            success = test_function()
            results[test_name] = "PASS" if success else "FAIL"
            logger.info(f"Result: {results[test_name]}")
        except Exception as error:
            logger.error(f"Test {test_name} crashed: {str(error)}")
            results[test_name] = "ERROR"

    # Summary
    logger.info("\n=== Test Results Summary ===")
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")

    passed = sum(1 for result in results.values() if result == "PASS")
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")

    return results


def demo_happy_path():
    """Demonstrate the happy path usage of the API."""
    logger.info("\n=== Bank Marketing Prediction API Happy Path Demo ===")

    # Check API status
    logger.info("1. Checking API status...")
    health_response = requests.get(f"{BASE_URL}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        logger.info(f"API Status: {health_data.get('status')}")
        logger.info(f"Model Loaded: {health_data.get('model_loaded')}")

    # Get model information
    logger.info("\n2. Retrieving model information...")
    model_response = requests.get(f"{BASE_URL}/model")
    if model_response.status_code == 200:
        model_info = model_response.json()
        logger.info(
            f"Model: {model_info.get('model_name')} v{model_info.get('model_version')}"
        )
        logger.info(f"Key Features: {model_info.get('top_features', [])[:3]}...")

    # Make predictions
    logger.info("\n3. Making sample predictions...")
    sample_cases = [
        {
            "case": "Young Student",
            "data": {
                "age": 23,
                "job": "student",
                "marital": "single",
                "education": "secondary",
                "default": "no",
                "balance": 200,
                "housing": "no",
                "loan": "yes",
                "contact": "cellular",
                "day": 10,
                "month": "apr",
                "duration": 150,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            },
        },
        {
            "case": "Middle-aged Manager",
            "data": {
                "age": 45,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 5000,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 20,
                "month": "jun",
                "duration": 400,
                "campaign": 3,
                "pdays": 180,
                "previous": 2,
                "poutcome": "success",
            },
        },
        {
            "case": "Retired Person",
            "data": {
                "age": 67,
                "job": "retired",
                "marital": "married",
                "education": "primary",
                "default": "no",
                "balance": 3500,
                "housing": "yes",
                "loan": "no",
                "contact": "telephone",
                "day": 5,
                "month": "nov",
                "duration": 800,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            },
        },
    ]

    for sample in sample_cases:
        logger.info(f"\nPredicting for: {sample['case']}")
        pred_response = requests.post(f"{BASE_URL}/predict", json=sample["data"])

        if pred_response.status_code == 200:
            prediction = pred_response.json()
            logger.info(f"  Input: {sample['data']}")
            logger.info(f"  Prediction: {prediction.get('prediction')} (0=No, 1=Yes)")
            logger.info(
                f"  Probability: {prediction.get('probability', 'N/A'):.4f}"
                if prediction.get("probability") is not None
                else "  Probability: N/A"
            )
        else:
            logger.warning(f"  Prediction failed: {pred_response.status_code}")


if __name__ == "__main__":
    # Run comprehensive tests
    test_results = run_comprehensive_tests()

    # Run happy path demo if basic tests pass
    if test_results.get("Root Endpoint") == "PASS":
        demo_happy_path()
    else:
        logger.error(
            "Basic connectivity failed. Check if the FastAPI server is running."
        )
        logger.info("Start the server with: uv run python src/serve/app.py")
        logger.info(
            "Or with uvicorn: uv run uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload"
        )
