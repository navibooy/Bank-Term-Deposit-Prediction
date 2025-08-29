#!/usr/bin/env python3
"""
Debug script to test MLflow connectivity from within container environment.
"""
import os
import requests
import mlflow
from mlflow.tracking import MlflowClient

def test_mlflow_connectivity():
    """Test MLflow connectivity with container networking."""
    print("🔍 Testing MLflow connectivity in container environment...")
    
    # Test different MLflow URIs
    uris_to_test = [
        "http://mlflow:5000",  # Container network
        "http://localhost:5000",  # Localhost
        "http://127.0.0.1:5000",  # Localhost IP
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")  # Environment
    ]
    
    for uri in uris_to_test:
        print(f"\n🧪 Testing URI: {uri}")
        
        # Test 1: HTTP connection
        try:
            response = requests.get(f"{uri}/health", timeout=5)
            print(f"   ✅ HTTP connection: {response.status_code}")
        except Exception as e:
            print(f"   ❌ HTTP connection failed: {e}")
            continue
        
        # Test 2: MLflow client
        try:
            mlflow.set_tracking_uri(uri)
            client = MlflowClient()
            experiments = client.search_experiments(max_results=1)
            print(f"   ✅ MLflow client works: Found {len(experiments)} experiments")
            
            # Test 3: Check for models
            try:
                models = client.search_registered_models()
                print(f"   📦 Found {len(models)} registered models")
                for model in models:
                    print(f"      - {model.name}")
            except Exception as model_error:
                print(f"   ⚠️  Model search failed: {model_error}")
                
            return uri  # Return working URI
            
        except Exception as e:
            print(f"   ❌ MLflow client failed: {e}")
    
    return None

def test_container_environment():
    """Test container environment setup."""
    print("🐳 Container Environment Check:")
    print(f"   MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
    print(f"   PYTHONPATH: {os.getenv('PYTHONPATH', 'Not set')}")
    print(f"   Current working dir: {os.getcwd()}")
    
    # Check if config file exists
    config_path = "config.yaml"
    if os.path.exists(config_path):
        print(f"   ✅ Config file exists: {config_path}")
    else:
        print(f"   ❌ Config file missing: {config_path}")
    
    # Check MLflow imports
    try:
        import mlflow
        print(f"   ✅ MLflow import: v{mlflow.__version__}")
    except Exception as e:
        print(f"   ❌ MLflow import failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Container MLflow Connectivity Debug")
    print("=" * 60)
    
    test_container_environment()
    working_uri = test_mlflow_connectivity()
    
    if working_uri:
        print(f"\n✅ MLflow is accessible at: {working_uri}")
        print("The FastAPI app should work with this URI.")
    else:
        print(f"\n❌ MLflow is not accessible from container!")
        print("Check that:")
        print("1. MLflow container is running: docker-compose ps")
        print("2. Network is properly configured")
        print("3. MLflow service is healthy")