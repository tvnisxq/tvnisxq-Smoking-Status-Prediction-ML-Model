import subprocess
import sys
import time
import os
import psutil
import signal

def kill_server_if_running():
    """Kill any process using port 8000"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Check if process has matching connection
            connections = psutil.Process(proc.info['pid']).connections()
            for conn in connections:
                if hasattr(conn, 'laddr') and conn.laddr.port == 8000:
                    print(f"Killing existing process on port 8000 (PID: {proc.info['pid']})")
                    os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def main():
    # First kill any existing server
    kill_server_if_running()
    
    # Start the FastAPI server
    print("\nStarting FastAPI server...")
    server = subprocess.Popen(
        ["uvicorn", "src.components.model_deployment:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to initialize...")
    time.sleep(5)
    
    # Run the tests
    print("\nRunning deployment tests...")
    test_result = subprocess.run(
        [sys.executable, "test_deployment.py"],
        capture_output=True,
        text=True
    )
    
    # Print test output
    print("\nTest Output:")
    print(test_result.stdout)
    if test_result.stderr:
        print("\nTest Errors:")
        print(test_result.stderr)
    
    # Stop the server
    print("\nShutting down server...")
    server.terminate()
    stdout, stderr = server.communicate()
    
    # Check for server errors
    if stderr:
        print("\nServer Errors:")
        print(stderr)
    
    return test_result.returncode == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {str(e)}")
        sys.exit(1)