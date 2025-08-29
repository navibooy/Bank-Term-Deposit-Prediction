#!/usr/bin/env python3
"""
Script to check MLflow registry status and optionally create a test model.
"""
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_mlflow_connection():
    """Check MLflow server connection and registry status."""
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        
        logger.info(f"Connecting to MLflow at: {mlflow_uri}")
        
        # Test connection by listing experiments
        experiments = client.search_experiments(max_results=5)
        logger.info(f"Found {len(experiments)} experiments")
        
        # List all registered models
        models = client.search_registered_models()
        logger.info(f"Found {len(models)} registered models:")
        
        for model in models:
            logger.info(f"  Model: {model.name}")
            # Get all versions for this model
            versions = client.get_latest_versions(model.name)
            for version in versions:
                logger.info(f"    Version {version.version} (Stage: {version.current_stage})")
        
        if not models:
            logger.warning("No registered models found!")
            logger.info("You need to run the training pipeline to create models.")
            logger.info("Or use the create_test_model() function below.")
        
        return True, client
        
    except Exception as e:
        logger.error(f"Failed to connect to MLflow: {e}")
        return False, None


def check_specific_model(client, model_name="champion", stage="Production"):
    """Check if a specific model exists in the registry."""
    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if latest_versions:
            version = latest_versions[0]
            logger.info(f"✅ Found model: {model_name}/{stage}")
            logger.info(f"   Version: {version.version}")
            logger.info(f"   Run ID: {version.run_id}")
            logger.info(f"   Created: {version.creation_timestamp}")
            return True
        else:
            logger.warning(f"❌ Model not found: {model_name}/{stage}")
            return False
    except Exception as e:
        logger.error(f"Error checking model {model_name}/{stage}: {e}")
        return False


def create_test_model(client, model_name="champion"):
    """Create a simple test model for development/testing purposes."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        logger.info(f"Creating test model: {model_name}")
        
        # Create dummy data
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Start MLflow run
        with mlflow.start_run(run_name="test_model_creation") as run:
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                input_example=pd.DataFrame(X[:5], columns=feature_names),
                registered_model_name=model_name
            )
            
            # Log some metrics
            mlflow.log_metrics({
                "roc_auc": 0.87,
                "f1_score": 0.82,
                "precision": 0.85,
                "recall": 0.79
            })
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": 10,
                "random_state": 42,
                "model_type": "RandomForest"
            })
            
            run_id = run.info.run_id
            
        logger.info(f"✅ Created model with run_id: {run_id}")
        
        # Promote to Production stage
        # Get the latest version number
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            version_number = latest_versions[0].version
            logger.info(f"Promoting version {version_number} to Production")
            
            client.transition_model_version_stage(
                name=model_name,
                version=version_number,
                stage="Production"
            )
            
            logger.info(f"✅ Model {model_name} version {version_number} promoted to Production")
            return True
        else:
            logger.error("Could not find created model version")
            return False
            
    except Exception as e:
        logger.error(f"Failed to create test model: {e}")
        return False


def main():
    """Main function to check MLflow status and optionally create test model."""
    print("=" * 60)
    print("MLflow Registry Status Check")
    print("=" * 60)
    
    # Check MLflow connection
    connected, client = check_mlflow_connection()
    if not connected:
        print("\n❌ Cannot connect to MLflow server.")
        print("Make sure MLflow server is running:")
        print("  uv run mlflow server --host 0.0.0.0 --port 5000")
        return
    
    print(f"\n✅ Connected to MLflow successfully")
    
    # Check for the specific model FastAPI expects
    model_exists = check_specific_model(client, "champion", "Production")
    
    if not model_exists:
        print(f"\n⚠️  Model 'champion/Production' not found!")
        print("This is needed for FastAPI to work properly.")
        
        response = input("\nWould you like to create a test model? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\nCreating test model...")
            success = create_test_model(client, "champion")
            if success:
                print("\n✅ Test model created successfully!")
                print("FastAPI should now be able to load the model.")
                
                # Verify the model was created
                check_specific_model(client, "champion", "Production")
            else:
                print("\n❌ Failed to create test model.")
        else:
            print("\nTo create a real model, run the training pipeline:")
            print("  # Start airflow and run training_pipeline DAG")
            print("  # Or run training scripts directly")
    else:
        print(f"\n✅ Model 'champion/Production' is available!")
        print("FastAPI should be able to load it successfully.")


if __name__ == "__main__":
    main()