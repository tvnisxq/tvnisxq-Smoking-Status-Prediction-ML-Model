import requests
import json
import time
import sys

def test_api():
    # Wait for server to start
    time.sleep(2)
    
    base_url = "http://localhost:8000"
    
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Health check failed: {str(e)}")
    
    print("\nTesting models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"Models response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Models endpoint failed: {str(e)}")
    
    # Test prediction
    test_data = {
        "height(cm)": 170.0,
        "weight(kg)": 70.0,
        "waist(cm)": 85.0,
        "eyesight(left)": 1.0,
        "eyesight(right)": 1.0,
        "age": 35.0,
        "ALT": 25.0,
        "Gtp": 30.0,
        "HDL": 50.0,
        "dental caries": 0,
        "fasting blood sugar": 90.0,
        "relaxation": 80.0,
        "serum creatinine": 1.0,
        "triglyceride": 150.0,
        "hemoglobin": 15.0,
        "systolic": 120.0
    }
    
    print("\nTesting prediction endpoints...")
    for model in ["ml_olympiad_improved_final", "archive_improved_final"]:
        try:
            response = requests.post(f"{base_url}/predict/{model}", json=test_data)
            print(f"\nPrediction with {model}:")
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"Prediction failed for {model}: {str(e)}")

if __name__ == "__main__":
    test_api()