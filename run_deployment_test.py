import subprocess
import time
import sys
import os
import signal
import psutil
import requests
import json
from pathlib import Path

def kill_process_by_port(port):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            for conns in psutil.Process(proc.info['pid']).connections(kind='inet'):
                if conns.laddr.port == port:
                    print(f"Killing process {proc.info['pid']} using port {port}")
                    os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def main():
    # Kill any process using port 8000
    kill_process_by_port(8000)
    
    # Start the FastAPI server
    print("Starting FastAPI server...")
    server_process = subprocess.Popen(
        ["uvicorn", "src.components.model_deployment:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    results = []
    try:
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
    
    finally:
        # Stop the server
        server_process.terminate()
        
        # Write results to file
        Path('deployment_test_results.json').write_text(
            json.dumps(results, indent=2)
        )
        
        # Print server output
        stdout, stderr = server_process.communicate()
        Path('server_output.log').write_text(
            f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        )

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        sys.exit(1)