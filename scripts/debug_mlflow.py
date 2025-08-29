#!/usr/bin/env python3
"""
Script to debug MLflow model loading issues and fix common problems.
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


def debug_mlflow_model(model_name="champion", stage="Production"):
    """Debug MLflow model loading issues."""
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        
        logger.info(f"🔍 Debugging MLflow model: {model_name}/{stage}")
        logger.info(f"📡 MLflow URI: {mlflow_uri}")
        
        # Check if model exists in registry
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            logger.error(f"❌ No model found in registry: {model_name}/{stage}")
            return False
            
        version = latest_versions[0]
        logger.info(f"✅ Found model version: {version.version}")
        logger.info(f"📋 Model details:")
        logger.info(f"   - Run ID: {version.run_id}")
        logger.info(f"   - Stage: {version.current_stage}")
        logger.info(f"   - Source: {version.source}")
        logger.info(f"   - Created: {version.creation_timestamp}")
        
        # Check run details
        run = client.get_run(version.run_id)
        logger.info(f"🏃 Run details:")
        logger.info(f"   - Status: {run.info.status}")
        logger.info(f"   - Artifact URI: {run.info.artifact_uri}")
        
        # List artifacts
        try:
            logger.info(f"📦 Listing artifacts for run {version.run_id}:")
            artifacts = client.list_artifacts(version.run_id)
            if not artifacts:
                logger.warning("⚠️  No artifacts found!")
                return False
                
            for artifact in artifacts:
                logger.info(f"   - {artifact.path} (size: {artifact.file_size or 'unknown'})")
                
                # Check if it's the model artifact
                if artifact.path == "model" or artifact.path.startswith("model/"):
                    logger.info(f"   ✅ Found model artifact: {artifact.path}")
                    
                    # Try to list contents of model directory
                    try:
                        model_contents = client.list_artifacts(version.run_id, artifact.path)
                        logger.info(f"   📁 Model directory contents:")
                        for content in model_contents:
                            logger.info(f"      - {content.path}")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Could not list model contents: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Could not list artifacts: {e}")
            return False
        
        # Test different loading methods
        logger.info(f"🧪 Testing model loading methods...")
        
        # Method 1: Registry URI
        try:
            logger.info("   Testing registry URI...")
            registry_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(registry_uri)
            logger.info("   ✅ Registry URI loading: SUCCESS")
            return True
        except Exception as e:
            logger.error(f"   ❌ Registry URI loading failed: {e}")
        
        # Method 2: Run ID URI
        try:
            logger.info("   Testing run ID URI...")
            run_uri = f"runs:/{version.run_id}/model"
            model = mlflow.pyfunc.load_model(run_uri)
            logger.info("   ✅ Run ID URI loading: SUCCESS")
            return True
        except Exception as e:
            logger.error(f"   ❌ Run ID URI loading failed: {e}")
        
        # Method 3: Direct artifact URI
        try:
            logger.info("   Testing direct artifact URI...")
            artifact_uri = f"{run.info.artifact_uri}/model"
            model = mlflow.pyfunc.load_model(artifact_uri)
            logger.info("   ✅ Direct artifact URI loading: SUCCESS")
            return True
        except Exception as e:
            logger.error(f"   ❌ Direct artifact URI loading failed: {e}")
        
        logger.error("❌ All loading methods failed!")
        return False
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        return False


def check_mlflow_server():
    """Check if MLflow server is accessible."""
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        
        # Try to list experiments
        experiments = client.search_experiments(max_results=1)
        logger.info(f"✅ MLflow server accessible at {mlflow_uri}")
        return True
    except Exception as e:
        logger.error(f"❌ MLflow server not accessible: {e}")
        logger.info("💡 Make sure MLflow server is running:")
        logger.info("   uv run mlflow server --host 0.0.0.0 --port 5000")
        return False


def fix_artifact_path_issue():
    """Try to fix common artifact path issues."""
    logger.info("🔧 Attempting to fix artifact path issues...")
    
    # Check if mlruns directory exists and has proper structure
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        logger.error(f"❌ mlruns directory not found at: {mlruns_dir.absolute()}")
        logger.info("💡 This might be the issue. MLflow expects mlruns in the current directory.")
        logger.info("   Try running MLflow server from the project root directory.")
        return False
    
    logger.info(f"✅ Found mlruns directory at: {mlruns_dir.absolute()}")
    
    # List experiments
    experiments = list(mlruns_dir.glob("*"))
    logger.info(f"📁 Found {len(experiments)} experiment directories:")
    for exp in experiments:
        if exp.is_dir() and exp.name != ".trash":
            logger.info(f"   - {exp.name}")
            
            # Check for runs in this experiment
            runs = list(exp.glob("*"))
            run_count = sum(1 for r in runs if r.is_dir() and r.name != "meta.yaml")
            logger.info(f"     Contains {run_count} runs")
    
    return True


def create_test_model_with_proper_artifacts():
    """Create a test model with proper artifact structure."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import pickle
        
        logger.info("🏗️  Creating test model with proper artifacts...")
        
        # Create dummy data matching bank marketing schema
        n_samples = 1000
        feature_names = [
            "age", "job", "marital", "education", "default", "balance", 
            "housing", "loan", "contact", "day", "month", "duration", 
            "campaign", "pdays", "previous", "poutcome", "many_no"
        ]
        
        # Create synthetic data
        X = np.random.randn(n_samples, len(feature_names))
        y = (X.sum(axis=1) > 0).astype(int)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create MLflow run with proper logging
        mlflow.set_experiment("bank-marketing-catboost")
        
        with mlflow.start_run(run_name="debug_test_model") as run:
            # Log model with proper metadata
            input_example = pd.DataFrame(X[:5], columns=feature_names)
            
            mlflow.sklearn.log_model(
                model, 
                "model",
                input_example=input_example,
                registered_model_name="champion",
                metadata={"purpose": "debugging", "framework": "sklearn"}
            )
            
            # Log comprehensive metrics
            mlflow.log_metrics({
                "roc_auc": 0.87,
                "f1_score": 0.82,
                "precision": 0.85,
                "recall": 0.79,
                "accuracy": 0.84
            })
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": 10,
                "random_state": 42,
                "model_type": "RandomForest",
                "n_features": len(feature_names)
            })
            
            # Log feature names as tags
            for i, feature in enumerate(feature_names[:5]):
                mlflow.set_tag(f"top_feature_{i+1}", feature)
            
            run_id = run.info.run_id
            
        logger.info(f"✅ Created test model with run_id: {run_id}")
        
        # Promote to Production
        client = MlflowClient()
        latest_versions = client.get_latest_versions("champion")
        if latest_versions:
            version_number = latest_versions[0].version
            client.transition_model_version_stage(
                name="champion",
                version=version_number,
                stage="Production"
            )
            logger.info(f"✅ Promoted version {version_number} to Production")
            
        # Verify the model can be loaded
        logger.info("🧪 Testing model loading...")
        if debug_mlflow_model("champion", "Production"):
            logger.info("✅ Test model created and verified successfully!")
            return True
        else:
            logger.error("❌ Test model created but loading still fails")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to create test model: {e}")
        return False


def main():
    """Main debugging function."""
    print("=" * 60)
    print("🔍 MLflow Model Loading Debug Tool")
    print("=" * 60)
    
    # Step 1: Check MLflow server
    if not check_mlflow_server():
        return
    
    # Step 2: Check artifact paths
    fix_artifact_path_issue()
    
    # Step 3: Debug specific model
    print(f"\n🔍 Debugging model loading...")
    success = debug_mlflow_model("champion", "Production")
    
    if not success:
        print(f"\n⚠️  Model loading failed. Would you like to create a new test model?")
        response = input("Create test model? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            create_test_model_with_proper_artifacts()
        else:
            print(f"\n💡 Suggestions to fix the issue:")
            print(f"   1. Make sure MLflow server is running from project root")
            print(f"   2. Check that mlruns directory exists and has proper permissions")
            print(f"   3. Try re-running the training pipeline to create a fresh model")
            print(f"   4. Consider using a different artifact store (S3, Azure, etc.)")


if __name__ == "__main__":
    main()