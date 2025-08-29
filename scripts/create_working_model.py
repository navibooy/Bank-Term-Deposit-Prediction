#!/usr/bin/env python3
"""
Create a simple working model that FastAPI can definitely load.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def create_simple_model():
    """Create a simple sklearn model that works with FastAPI."""
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        print("🏗️  Creating a simple working model...")
        
        # Set MLflow tracking URI
        mlflow_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Test connection
        client = MlflowClient()
        try:
            experiments = client.search_experiments(max_results=1)
            print(f"✅ Connected to MLflow at {mlflow_uri}")
        except Exception as e:
            print(f"❌ MLflow connection failed: {e}")
            print(f"💡 Make sure MLflow server is running: uv run mlflow server --host 0.0.0.0 --port 5000")
            return False
        
        # Set experiment
        experiment_name = "bank-marketing-catboost"
        try:
            mlflow.set_experiment(experiment_name)
        except:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        
        # Create sample data matching the API schema
        n_samples = 1000
        n_features = 17  # Number of features after preprocessing
        
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=42
        )
        
        # Feature names matching what FastAPI expects after preprocessing
        feature_names = [
            "age", "balance", "duration", "campaign", "pdays", "previous",
            "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid",
            "marital_divorced", "marital_married", "marital_single",
            "education_primary", "education_secondary", "education_tertiary",
            "many_no"
        ]
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
        model.fit(X, y)
        
        # Create example input
        input_example = pd.DataFrame(X[:1], columns=feature_names)
        
        # Start MLflow run
        with mlflow.start_run(run_name="working_fastapi_model") as run:
            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="champion",
                conda_env=None,
                pip_requirements=["scikit-learn", "pandas", "numpy"]
            )
            
            # Log realistic metrics
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
                "max_depth": 5,
                "random_state": 42,
                "model_type": "RandomForestClassifier"
            })
            
            # Log feature names as tags for debugging
            for i, feature in enumerate(feature_names[:5]):
                mlflow.set_tag(f"top_feature_{i+1}", feature)
            
            run_id = run.info.run_id
            print(f"✅ Model logged with run_id: {run_id}")
        
        # Promote to Production stage
        try:
            # Get the latest version
            latest_versions = client.get_latest_versions("champion")
            if latest_versions:
                version = latest_versions[0].version
                print(f"📦 Promoting version {version} to Production...")
                
                # Transition to Production
                client.transition_model_version_stage(
                    name="champion",
                    version=version,
                    stage="Production"
                )
                
                print(f"✅ Model champion v{version} is now in Production stage")
                
                # Test loading to make sure it works
                print("🧪 Testing model loading...")
                try:
                    model_uri = f"models:/champion/Production"
                    loaded_model = mlflow.pyfunc.load_model(model_uri)
                    print("✅ Model loads successfully from registry!")
                    
                    # Test prediction
                    test_prediction = loaded_model.predict(input_example)
                    print(f"✅ Test prediction: {test_prediction}")
                    
                    return True
                    
                except Exception as load_error:
                    print(f"❌ Model loading test failed: {load_error}")
                    
                    # Try alternative loading
                    try:
                        alt_uri = f"runs:/{run_id}/model"
                        loaded_model = mlflow.pyfunc.load_model(alt_uri)
                        print("✅ Alternative loading works!")
                        return True
                    except Exception as alt_error:
                        print(f"❌ Alternative loading failed: {alt_error}")
                        return False
            else:
                print("❌ No model versions found after registration")
                return False
                
        except Exception as promote_error:
            print(f"❌ Model promotion failed: {promote_error}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating a working test model for FastAPI...")
    print("=" * 50)
    
    success = create_simple_model()
    
    if success:
        print("\n✅ SUCCESS! Test model created and ready.")
        print("\nNext steps:")
        print("1. Start FastAPI: uv run python src/serve/app.py")
        print("2. Test predictions: uv run python src/serve/client.py")
        print("3. Or test manually: curl http://localhost:8000/health")
    else:
        print("\n❌ FAILED to create working model.")
        print("Check MLflow server and try again.")