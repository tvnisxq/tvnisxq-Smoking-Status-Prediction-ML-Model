import os
import json
import sys
import time
import requests
from pathlib import Path
import subprocess
import logging
from datetime import datetime
import psutil
import signal

# Define log directories
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / 'logs'
TEST_LOG_DIR = LOG_DIR / 'tests'
SERVER_LOG_DIR = LOG_DIR / 'server'
RESULTS_DIR = LOG_DIR / 'results'

# Create log directories
LOG_DIR.mkdir(exist_ok=True)
TEST_LOG_DIR.mkdir(exist_ok=True)
SERVER_LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TEST_LOG_DIR / f'deployment_test_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def kill_process_by_port(port):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            for conns in psutil.Process(proc.info['pid']).connections(kind='inet'):
                if conns.laddr.port == port:
                    logging.info(f"Killing process {proc.info['pid']} using port {port}")
                    os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def main():
    """Run deployment tests"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    try:
        # Kill any process using port 8000
        kill_process_by_port(8000)
        
        # Start the server
        logging.info("Starting FastAPI server...")
        server_process = subprocess.Popen(
            ["uvicorn", "src.components.model_deployment:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(5)
        logging.info("Server started. Running tests...")
        
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health")
        results.append({
            "test": "health_check",
            "status": "success" if health_response.status_code == 200 else "failed",
            "response": health_response.json() if health_response.status_code == 200 else str(health_response.status_code)
        })
        
        # Test models endpoint
        models_response = requests.get("http://localhost:8000/models")
        results.append({
            "test": "list_models",
            "status": "success" if models_response.status_code == 200 else "failed",
            "response": models_response.json() if models_response.status_code == 200 else str(models_response.status_code)
        })
        
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
        
        for model in ["ml_olympiad_improved_final", "archive_improved_final"]:
            prediction_response = requests.post(
                f"http://localhost:8000/predict/{model}",
                json=test_data
            )
            results.append({
                "test": f"prediction_{model}",
                "status": "success" if prediction_response.status_code == 200 else "failed",
                "response": prediction_response.json() if prediction_response.status_code == 200 else str(prediction_response.status_code)
            })
    
    except Exception as e:
        results.append({
            "test": "overall",
            "status": "failed",
            "error": str(e)
        })
        logging.error(f"Test failed with error: {str(e)}")
    
    finally:
        # Stop the server
        server_process.terminate()
        
        # Save test results
        results_file = RESULTS_DIR / f'deployment_test_results_{timestamp}.json'
        results_file.write_text(json.dumps(results, indent=2))
        
        # Save server output
        stdout, stderr = server_process.communicate()
        server_log = SERVER_LOG_DIR / f'server_output_{timestamp}.log'
        with open(server_log, 'w') as f:
            f.write(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
        
        logging.info(f"Test results saved to {results_file}")
        logging.info(f"Server output saved to {server_log}")

if __name__ == "__main__":
    main()