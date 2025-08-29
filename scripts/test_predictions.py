#!/usr/bin/env python3
"""
Simple script to test FastAPI predictions with various customer profiles.
"""
import requests
import json

API_URL = "http://localhost:8000"

def make_prediction(customer_data, customer_name=""):
    """Make a prediction for a customer."""
    try:
        print(f"\n🧪 Testing prediction for: {customer_name}")
        print(f"📋 Input data: {json.dumps(customer_data, indent=2)}")
        
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction')
            probability = result.get('probability', 0)
            
            print(f"✅ Prediction: {'Will Subscribe' if prediction == 1 else 'Will NOT Subscribe'}")
            print(f"🎯 Probability: {probability:.4f}")
            print(f"🏷️  Model Version: {result.get('model_version', 'unknown')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_various_customers():
    """Test predictions for different customer profiles."""
    
    customers = [
        {
            "name": "Young Student (Low Income)",
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
                "poutcome": "unknown"
            }
        },
        {
            "name": "Middle-aged Technician", 
            "data": {
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
                "poutcome": "unknown"
            }
        },
        {
            "name": "Wealthy Manager (High Income)",
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
                "poutcome": "success"
            }
        },
        {
            "name": "Retired Person",
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
                "poutcome": "unknown"
            }
        }
    ]
    
    print("🏦 Bank Term Deposit Prediction Testing")
    print("=" * 50)
    
    # Check API health first
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"🟢 API Status: {health_data.get('status', 'unknown')}")
            print(f"📦 Model Loaded: {health_data.get('checks', {}).get('model_loaded', False)}")
        else:
            print(f"🔴 API Health Check Failed: {health_response.status_code}")
            return
    except Exception as e:
        print(f"🔴 Cannot connect to API: {e}")
        print(f"💡 Make sure FastAPI is running at {API_URL}")
        return
    
    # Test predictions
    successful_predictions = 0
    for customer in customers:
        if make_prediction(customer["data"], customer["name"]):
            successful_predictions += 1
    
    print(f"\n📊 Results Summary:")
    print(f"   ✅ Successful predictions: {successful_predictions}/{len(customers)}")
    
    if successful_predictions == len(customers):
        print(f"   🎉 All predictions successful!")
    elif successful_predictions > 0:
        print(f"   ⚠️  Some predictions failed - check API logs")
    else:
        print(f"   ❌ All predictions failed - check API status")

if __name__ == "__main__":
    test_various_customers()