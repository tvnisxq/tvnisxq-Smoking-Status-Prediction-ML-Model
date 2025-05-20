import logging
import requests
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from requests.exceptions import ConnectionError, Timeout
import unittest
import pandas as pd
from src.components.model_deployment import ModelDeployment
from src.components.feature_engineering import FeatureEngineer

# Define log directories
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / 'logs'
TEST_LOG_DIR = LOG_DIR / 'tests'
DEPLOYMENT_LOG_DIR = LOG_DIR / 'deployment'
ERROR_LOG_DIR = LOG_DIR / 'errors'

# Create log directories
LOG_DIR.mkdir(exist_ok=True)
TEST_LOG_DIR.mkdir(exist_ok=True)
DEPLOYMENT_LOG_DIR.mkdir(exist_ok=True)
ERROR_LOG_DIR.mkdir(exist_ok=True)

# Create a custom formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Configure file handlers
test_handler = logging.FileHandler(TEST_LOG_DIR / f'test_{datetime.now().strftime("%Y%m%d")}.log')
test_handler.setFormatter(formatter)

deployment_handler = logging.FileHandler(DEPLOYMENT_LOG_DIR / f'deployment_{datetime.now().strftime("%Y%m%d")}.log')
deployment_handler.setFormatter(formatter)

error_handler = logging.FileHandler(ERROR_LOG_DIR / f'error_{datetime.now().strftime("%Y%m%d")}.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

# Configure console handler with immediate flush
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(test_handler)
root_logger.addHandler(deployment_handler)
root_logger.addHandler(error_handler)
root_logger.addHandler(console_handler)

def write_error_to_file(error_msg):
    """Write error message to a timestamped file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_file = ERROR_LOG_DIR / f"error_{timestamp}.txt"
    with open(error_file, "w") as f:
        f.write(f"=== Error Log {timestamp} ===\n")
        f.write(error_msg)
        f.write("\n=== End Error Log ===\n")
    print(f"Error details written to {error_file}")

def wait_for_server(timeout=30):
    """Wait for server to be ready"""
    import time
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return True
        except (ConnectionError, Timeout):
            time.sleep(1)
    return False

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        logging.info("Sending health check request...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code != 200:
            error_msg = f"Health check failed with status {response.status_code}: {response.text}"
            write_error_to_file(error_msg)
            response.raise_for_status()
            
        data = response.json()
        logging.info(f"Health check response: {json.dumps(data, indent=2)}")
        
        assert "status" in data, f"Missing 'status' in response: {data}"
        assert data["status"] == "healthy", f"Unhealthy status: {data['status']}"
        assert "models_loaded" in data, f"Missing 'models_loaded' in response: {data}"
        return data["models_loaded"]
        
    except Exception as e:
        error_msg = f"""
Health check failed:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
        write_error_to_file(error_msg)
        raise

def test_model_prediction(model_name: str):
    """Test model prediction endpoint"""
    try:
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
        
        logging.info(f"Sending prediction request to model {model_name}...")
        response = requests.post(
            f"http://localhost:8000/predict/{model_name}",
            json=test_data,
            timeout=10
        )
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            write_error_to_file(error_msg)
            response.raise_for_status()
            
        prediction = response.json()
        logging.info(f"Raw response for {model_name}: {json.dumps(prediction, indent=2)}")
        
        assert "prediction" in prediction, f"Missing 'prediction' in response: {prediction}"
        assert "confidence" in prediction, f"Missing 'confidence' in response: {prediction}"
        assert "model_used" in prediction, f"Missing 'model_used' in response: {prediction}"
        return prediction
        
    except Exception as e:
        error_msg = f"""
Error testing model {model_name}:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
        write_error_to_file(error_msg)
        raise

def main():
    """Main test function"""
    logging.info("Starting deployment tests...")
    
    try:
        # Wait for server to be ready
        logging.info("Checking if server is ready...")
        if not wait_for_server():
            error_msg = "Server not responding after 30 seconds"
            write_error_to_file(error_msg)
            return False
            
        # Test health endpoint
        try:
            logging.info("Testing health endpoint...")
            loaded_models = test_health_endpoint()
            logging.info(f"Health check passed. Models loaded: {loaded_models}")
        except Exception as e:
            error_msg = f"""
Health endpoint test failed:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
            write_error_to_file(error_msg)
            return False
        
        # Test each model
        for model in loaded_models:
            try:
                logging.info(f"\nTesting model: {model}")
                prediction = test_model_prediction(model)
                logging.info(f"Prediction test passed for {model}:")
                logging.info(json.dumps(prediction, indent=2))
            except Exception as e:
                error_msg = f"""
Model test failed for {model}:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
                write_error_to_file(error_msg)
                return False
            
        logging.info("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        error_msg = f"""
Unexpected error in main:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
        write_error_to_file(error_msg)
        return False

class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        self.deployment = ModelDeployment()
        self.sample_input = {
            "age": 25,
            "height": 170,
            "weight": 70,
            "waist": 80,
            "eyesight_left": 1.0,
            "eyesight_right": 1.0,
            "hearing_left": 1,
            "hearing_right": 1,
            "systolic": 120,
            "relaxation": 80,
            "fasting_blood_sugar": 90,
            "cholesterol": 200,
            "triglyceride": 150,
            "hdl": 50,
            "ldl": 100,
            "hemoglobin": 14,
            "urine_protein": 1,
            "serum_creatinine": 0.9,
            "ast": 25,
            "alt": 25,
            "gtp": 30,
            "dental_caries": 0
        }

    async def test_prediction_pipeline(self):
        result = await self.deployment.predict(self.sample_input)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'success')
        self.assertIsInstance(result['prediction'], int)
        self.assertIsInstance(result['probability'], float)

    def test_feature_engineering(self):
        df = pd.DataFrame([self.sample_input])
        engineered_df = self.deployment.feature_engineer.transform(df)
        
        # Verify that all expected engineered features are present
        expected_features = ['bmi', 'blood_pressure_category', 'cholesterol_ratio']
        for feature in expected_features:
            self.assertIn(feature, engineered_df.columns)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        error_msg = f"""
Critical error:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
        write_error_to_file(error_msg)
        sys.exit(1)