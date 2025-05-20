@echo off
echo Killing any existing Python processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak

echo Starting FastAPI server...
start "FastAPI Server" cmd /k "uvicorn src.components.model_deployment:app --host 0.0.0.0 --port 8000 --log-level debug"
timeout /t 5 /nobreak

echo Running tests...
python test_deployment.py
echo Tests completed.

echo Press any key to stop the server...
pause
taskkill /F /IM python.exe