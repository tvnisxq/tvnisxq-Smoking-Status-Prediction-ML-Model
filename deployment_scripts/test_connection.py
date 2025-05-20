import socket
import requests
import sys
import os
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('connection_test.log'),
        logging.StreamHandler()
    ]
)

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"Port {port} is open on {host}")
            return True
        else:
            print(f"Port {port} is closed on {host}")
            return False
    except Exception as e:
        print(f"Error checking port: {str(e)}")
        return False
    finally:
        sock.close()

def test_connection(url):
    print(f"\nTesting connection to {url}")
    
    # Parse URL
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 80
    
    # Check if port is open
    if not check_port(host, port):
        print("Port is not accessible. Server may not be running.")
        return
    
    # Try HTTP connection
    try:
        response = requests.get(url, timeout=5)
        print(f"HTTP Status: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    urls = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8000/health",
        "http://localhost:8000/health"
    ]
    
    print("Starting connection tests...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    for url in urls:
        test_connection(url)