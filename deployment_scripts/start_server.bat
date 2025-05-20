


















@echo off
echo Starting Smoking Status Prediction API server...
echo.

REM Activate the virtual environment if needed
REM call Y:\SmokingML V2\SmokeML_v2_venv\Scripts\activate

REM Start the FastAPI server
python -m uvicorn src.components.model_deployment:app --reload --host 0.0.0.0 --port 8000

echo.
echo Server stopped.