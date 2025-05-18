@echo off
echo Starting FastAPI server...
echo Press Ctrl+C to stop the server

set PYTHONUNBUFFERED=1

REM Kill any existing process on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000"') do taskkill /F /PID %%a 2>nul

REM Start the server with unbuffered output
python -u -m uvicorn src.components.model_deployment:app --host 127.0.0.1 --port 8000 --log-level debug

echo Server stopped.