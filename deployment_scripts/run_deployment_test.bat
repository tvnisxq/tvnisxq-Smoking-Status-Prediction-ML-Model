@echo off
echo Starting FastAPI server...
start /B cmd /k "uvicorn src.components.model_deployment:app --reload --host 0.0.0.0 --port 8000 --log-level debug"

echo Waiting for server to start...
timeout /t 5 /nobreak

echo Running tests...
python tests/test_deployment.py

echo Done.