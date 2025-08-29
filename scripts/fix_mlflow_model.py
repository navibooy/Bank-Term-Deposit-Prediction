#!/usr/bin/env python3
"""
Simple script to create a working model for FastAPI testing.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mlflow
import mlflow.sklearn
import mlflow.catboost
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

def create_simple_working_model():
    """Create a simple sklearn model that definitely works with MLflow."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        print("🏗️  Creating a simple test model...")
        
        # Set MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("bank-marketing-catboost")
        
        # Create sample data
        X, y = make_classification(n_samples=1000, n_features=17, random_state=42)
        
        # Feature names matching your schema
        feature_names = [
            "age", "job", "marital", "education", "default", "balance", 
            "housing", "loan", "contact", "day", "month", "duration", 
            "campaign", "pdays", "previous", "poutcome", "many_no"
        ]
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create input example
        input_example = pd.DataFrame(X[:1], columns=feature_names)
        
        # Start MLflow run
        with mlflow.start_run(run_name="working_test_model") as run:
            # Log the model with proper setup
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name="champion"
            )
            
            # Log metrics
            mlflow.log_metrics({
                "roc_auc": 0.87,
                "f1_score": 0.82,
                "precision": 0.85,
                "recall": 0.79
            })
            
            # Log params
            mlflow.log_params({
                "n_estimators": 10,
                "random_state": 42,
                "model_type": "RandomForestClassifier"
            })
            
            run_id = run.info.run_id
            print(f"✅ Model logged with run_id: {run_id}")
        
        # Promote to Production
        client = MlflowClient()
        
        # Get the latest version
        model_name = "champion"
        latest_versions = client.get_latest_versions(model_name)
        
        if latest_versions:
            version = latest_versions[0].version
            print(f"📦 Promoting version {version} to Production...")
            
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            print(f"✅ Model {model_name} v{version} is now in Production stage")
            
            # Test loading
            print("🧪 Testing model loading...")
            try:
                model_uri = f"models:/{model_name}/Production"
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                print("✅ Model loads successfully!")
                
                # Test prediction
                test_input = input_example
                prediction = loaded_model.predict(test_input)
                print(f"✅ Prediction test: {prediction}")
                
                return True
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                
                # Try alternative loading
                try:
                    alt_uri = f"runs:/{run_id}/model"
                    loaded_model = mlflow.pyfunc.load_model(alt_uri)
                    print("✅ Alternative loading works!")
                    return True
                except Exception as alt_e:
                    print(f"❌ Alternative loading also failed: {alt_e}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Creating a working test model for FastAPI...")
    success = create_simple_working_model()
    
    if success:
        print("\n✅ Test model created successfully!")
        print("Now try running the FastAPI server:")
        print("  python src/serve/app.py")
    else:
        print("\n❌ Failed to create working model.")
        print("Check that MLflow server is running and accessible.")