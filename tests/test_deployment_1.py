import logging
import requests
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_and_log(message, error=False):
    """Print to console and write to log file immediately"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Use ASCII symbols instead of unicode
    message = message.replace("✓", "[PASS]").replace("✗", "[FAIL]")
    output = f"[{timestamp}] {'ERROR: ' if error else ''}{message}"
    print(output, flush=True)
    with open("deployment_results.log", "a", encoding='utf-8') as f:
        f.write(output + "\n")
        f.flush()

def test_endpoint(method, url, json_data=None, max_retries=3, retry_delay=2):
    """Test an endpoint with retries"""
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=json_data, timeout=10)
                
            response_text = response.text
            if response.ok:
                return response.json()
            else:
                print_and_log(f"Request failed with status {response.status_code}: {response_text}", error=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
                
        except requests.exceptions.ConnectionError:
            print_and_log(f"Connection failed on attempt {attempt + 1}/{max_retries}", error=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise
        except Exception as e:
            print_and_log(f"Unexpected error: {str(e)}", error=True)
            raise

def test_deployment():
    """Test deployment endpoints"""
    base_url = "http://localhost:8000"
    all_tests_passed = True
    
    # Test health endpoint
    try:
        print_and_log("Testing health endpoint...")
        health_data = test_endpoint('get', f'{base_url}/health')
        if health_data and health_data.get('status') == 'healthy':
            print_and_log("[PASS] Health check passed")
            print_and_log(f"Loaded models: {health_data.get('models_loaded', [])}")
        else:
            print_and_log("[FAIL] Health check failed", error=True)
            all_tests_passed = False
    except Exception as e:
        print_and_log(f"Health check error: {str(e)}", error=True)
        traceback.print_exc()
        all_tests_passed = False
    
    # Test models endpoint
    try:
        print_and_log("\nTesting models endpoint...")
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        logger.info(f"Models endpoint successful: {response.json()}")
    except Exception as e:
        logger.error(f"Models endpoint failed: {str(e)}")
        all_tests_passed = False

    # Test prediction endpoint with complete test data
    test_data = {
        "height(cm)": 170.0,
        "weight(kg)": 70.0,
        "waist(cm)": 85.0,
        "eyesight(left)": 1.0,
        "eyesight(right)": 1.0,
        "age": 35.0,
        "ALT": 25.0,
        "AST": 20.0,
        "Gtp": 30.0,
        "HDL": 50.0,
        "LDL": 100.0,
        "Cholesterol": 180.0,
        "dental caries": 0,
        "fasting blood sugar": 90.0,
        "relaxation": 80.0,
        "serum creatinine": 1.0,
        "triglyceride": 150.0,
        "hemoglobin": 15.0,
        "systolic": 120.0
    }

    models = ["ml_olympiad_improved_final", "archive_improved_final"]
    
    for model in models:
        try:
            print_and_log(f"\nTesting {model}...")
            prediction = test_endpoint(
                'post', 
                f'{base_url}/predict/{model}', 
                test_data
            )
            if prediction:
                print_and_log("✓ Prediction successful")
                print_and_log(f"Result: {json.dumps(prediction, indent=2)}")
            else:
                print_and_log(f"✗ Prediction failed for {model}", error=True)
                all_tests_passed = False
        except Exception as e:
            print_and_log(f"Prediction error for {model}: {str(e)}", error=True)
            traceback.print_exc()
            all_tests_passed = False
    
    # Print final status
    if all_tests_passed:
        print_and_log("\n✓ All tests passed successfully!")
    else:
        print_and_log("\n✗ Some tests failed. Check the logs for details.", error=True)
    
    return all_tests_passed

if __name__ == "__main__":
    logger.info("Starting deployment test...")
    
    # Give the server time to start if it's not already running
    for i in range(3):
        try:
            requests.get("http://localhost:8000/health")
            break
        except:
            if i == 2:
                logger.error("Server not responding after multiple attempts")
                sys.exit(1)
            logger.info("Waiting for server to start...")
            time.sleep(2)
    
    try:
        success = test_deployment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_and_log("\nTests interrupted by user.", error=True)
        sys.exit(1)
    except Exception as e:
        print_and_log(f"\nUnexpected error: {str(e)}", error=True)
        traceback.print_exc()
        sys.exit(1)