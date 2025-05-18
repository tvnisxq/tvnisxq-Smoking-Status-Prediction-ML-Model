from fastapi import FastAPI, HTTPException, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi.openapi.utils import get_openapi
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import PolynomialFeatures
from typing import Optional
from contextlib import asynccontextmanager
import socket
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse


# Configure logging to both file and console with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define model path and dictionary to hold loaded models
MODEL_PATH = os.getenv("MODEL_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))
logger.info(f"Using model path: {MODEL_PATH}")
models = {}

# Define best models to be loaded for deployment
BEST_MODELS = {
    "ml_olympiad_improved_final": "ML Olympiad – Improved XGBoost",
    "archive_improved_final": "Archive – Improved Ensemble"
}

# Define lifespan to load only the best models
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(f"Starting model loading from {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model directory not found at {MODEL_PATH}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
        logger.info(f"Found model files: {model_files}")
        
        if not model_files:
            error_msg = f"No .pkl model files found in {MODEL_PATH}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            if model_name in BEST_MODELS:
                model_path = os.path.join(MODEL_PATH, model_file)
                try:
                    logger.info(f"Loading model {model_name} from {model_path}")
                    model_artifacts = joblib.load(model_path)
                    models[model_name] = model_artifacts['model']
                    logger.info(f"Successfully loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {str(e)}")
                    raise
                    
        if not models:
            error_msg = f"No best models found for deployment in {MODEL_PATH}. Expected models: {list(BEST_MODELS.keys())}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise e
    yield
    # Cleanup
    logger.info("Cleaning up models")
    models.clear()

# Initialize FastAPI app with lifespan and custom startup message
app = FastAPI(
    lifespan=lifespan,
    title="Smoking Status Prediction API",
    description="API for predicting smoking status using machine learning models",
    version="2.0.0",
    docs_url=None,
    redoc_url=None
)

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure more detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_debug.log')
    ]
)

# Add socket info logging
def get_ip():
    try:
        # Get all network interfaces
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return f"Hostname: {hostname}, Local IP: {local_ip}"
    except Exception as e:
        return f"Could not determine IP: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Print detailed access information on startup"""
    logger.info("=== Server Starting ===")
    logger.info(get_ip())
    logger.info("You can access the API at:")
    logger.info("    http://127.0.0.1:8000")
    logger.info("    http://localhost:8000")
    logger.info("API documentation available at:")
    logger.info("    http://127.0.0.1:8000/docs")
    logger.info("    http://localhost:8000/docs")
    logger.info("Try both URLs if one doesn't work")

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Smoking Status Prediction API",
        version="2.0.0",
        description="**API for predicting smoking status using best-performing or ensemble ML models**",
        routes=app.routes,
    )

    # Define tags with descriptions and colors
    openapi_schema["tags"] = [
        {
            "name": "Root",
            "description": "**Root endpoint operations**",
            "x-tag-style": {"background-color": "#FFEB3B"}
        },
        {
            "name": "Models",
            "description": "**Model listing operations**",
            "x-tag-style": {"background-color": "#FF69B4"}
        },
        {
            "name": "Health",
            "description": "**Health check operations**",
            "x-tag-style": {"background-color": "#4CAF50"}
        },
        {
            "name": "Predictions",
            "description": "**Smoking status prediction operations**",
            "x-tag-style": {"background-color": "#2196F3"}
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Define input schema
class SmokingPredictionInput(BaseModel):
    height_cm: float = Field(..., alias="height(cm)")
    weight_kg: float = Field(..., alias="weight(kg)")
    waist_cm: float = Field(..., alias="waist(cm)")
    age: float
    ALT: float
    Gtp: float
    HDL: float
    LDL: float = Field(0.0)
    Cholesterol: float = Field(0.0)
    systolic: float
    relaxation: float
    hemoglobin: float
    serum_creatinine: float = Field(..., alias="serum creatinine")
    triglyceride: float
    AST: Optional[float] = Field(0.0)
    dental_caries: Optional[int] = Field(0, alias="dental caries")
    eyesight_right: Optional[float] = Field(0.0, alias="eyesight(right)")
    eyesight_left: Optional[float] = Field(0.0, alias="eyesight(left)")
    fasting_blood_sugar: Optional[float] = Field(0.0, alias="fasting blood sugar")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with detailed information"""
    network_info = get_ip()
    logger.info(f"Root endpoint accessed. {network_info}")
    return {
        "message": "Welcome to the Enhanced Smoking Prediction API",
        "models_available": list(models.keys()),
        "model_path": MODEL_PATH,
        "status": "healthy",
        "network_info": network_info,
        "docs_url": "/docs"
    }

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }

# Endpoint to list models
@app.get("/models", tags=["Models"])
async def list_models():
    logger.info("Models endpoint accessed")
    return {
        "available_models": BEST_MODELS,
        "loaded_models": list(models.keys()),
        "total": len(BEST_MODELS),
        "model_path": MODEL_PATH
    }

# Prediction endpoint
@app.post("/predict/{model_name}", tags=["Predictions"])
async def predict(
    model_name: str = Path(
        ...,
        description="Available models: ml_olympiad_improved_final, archive_improved_final"
    ),
    input_data: SmokingPredictionInput = Body(...)
):
    logger.info(f"Prediction requested for model: {model_name}")
    try:
        # Clean up model name by removing any extra whitespace
        model_name = model_name.strip()
        
        if model_name not in models:
            error_msg = f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail={"error": error_msg})

        data = pd.DataFrame([input_data.dict(by_alias=True)])
        logger.debug(f"Input data: {data.to_dict()}")
        
        # Calculate advanced features
        try:
            # Basic health indicators
            data['bmi'] = data['weight(kg)'] / ((data['height(cm)']/100) ** 2)
            data['liver_function'] = (data['AST'] + data['ALT'] + data['Gtp']) / 3
            data['cardiovascular_risk'] = (data['systolic'] * data['triglyceride']) / (data['HDL'] + 1)
            data['metabolic_index'] = data['fasting blood sugar'] * data['bmi'] / (data['HDL'] + 1)
            
            # Calculate polynomial features
            key_features = ['bmi', 'liver_function', 'cardiovascular_risk', 'metabolic_index']
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(data[key_features])
            poly_names = [f'health_poly_{i}' for i in range(poly_features.shape[1])]
            data[poly_names] = poly_features
            
            # Additional ratios and indicators
            data['hdl_ldl_ratio'] = data['HDL'] / (data['LDL'] + 1)
            data['ast_alt_ratio'] = data['AST'] / (data['ALT'] + 1)
            data['bp_ratio'] = data['systolic'] / (data['relaxation'] + 1)
            data['age_health_index'] = data['age'] * data['hemoglobin'] / (data['liver_function'] + 1)
            
            logger.debug("Advanced features calculated successfully")
        except Exception as e:
            error_msg = f"Error calculating advanced features: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail={"error": error_msg})

        # Select correct features based on model
        if model_name == 'ml_olympiad_improved_final':
            required_features = [
                "age", "height(cm)", "weight(kg)", "systolic", "relaxation",
                "Cholesterol", "triglyceride", "HDL", "LDL", "hemoglobin",
                "serum creatinine", "AST", "ALT", "Gtp", "dental caries",
                "health_poly_0", "health_poly_1", "health_poly_4", "health_poly_13",
                "bmi", "liver_function", "hdl_ldl_ratio", "ast_alt_ratio"
            ]
        else:  # archive_improved_final
            required_features = [
                "age", "height(cm)", "weight(kg)", "waist(cm)", "systolic",
                "relaxation", "fasting blood sugar", "triglyceride", "HDL",
                "LDL", "hemoglobin", "serum creatinine", "ALT", "Gtp",
                "dental caries", "health_poly_0", "health_poly_4", "health_poly_5",
                "bmi", "liver_function", "hdl_ldl_ratio", "ast_alt_ratio",
                "cardiovascular_risk", "metabolic_index", "bp_ratio"
            ]

        # Fill missing features with 0
        for feature in required_features:
            if feature not in data.columns:
                data[feature] = 0.0

        # Select only required features in correct order
        data = data[required_features]
        
        model = models[model_name]
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        confidence = float(max(probabilities))
        
        result = {
            "model_used": BEST_MODELS[model_name],
            "prediction": int(prediction),
            "label": "Smoker" if prediction == 1 else "Non-smoker",
            "confidence": f"{confidence:.2%}",
            "model_type": "XGBoost" if model_name == "ml_olympiad_improved_final" else "Ensemble",
            "features_used": required_features
        }
        logger.info(f"Prediction successful: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail={"error": error_msg})

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <link rel="shortcut icon" href="/favicon.ico">
        <title>Smoking Status Prediction API - Swagger UI</title>
        <style>
            /* Hide the filter input */
            .swagger-ui .filter-container {
                display: none !important;
            }
            
            /* API Description and Tag Description Styling */
            .title, .description, .opblock-tag-section h3 span, .opblock-tag-section .markdown p {
                font-weight: bold !important;
                font-size: calc(100% + 2pt) !important;
            }
            .info__title {
                font-weight: bold !important;
                font-size: calc(100% + 4pt) !important;
            }

            /* Method button styling */
            .swagger-ui .opblock-summary-method {
                min-width: 80px !important;
                text-align: center !important;
                border-radius: 3px !important;
                padding: 6px 15px !important;
            }

            /* Root endpoint (Yellow) */
            .swagger-ui #operations-Root-get .opblock-summary-method,
            .swagger-ui #operations-Root-get .btn,
            .swagger-ui #operations-Root-get .execute,
            .swagger-ui #operations-Root-get .try-out__btn {
                background: #FFD700 !important;
                border-color: #FFD700 !important;
                color: #000000 !important;
            }
            .swagger-ui #operations-Root-get.is-open .opblock-summary {
                border-color: #FFD700 !important;
            }

            /* Models endpoint (Purple) */
            .swagger-ui #operations-Models-get .opblock-summary-method,
            .swagger-ui #operations-Models-get .btn,
            .swagger-ui #operations-Models-get .execute,
            .swagger-ui #operations-Models-get .try-out__btn {
                background: #9B59B6 !important;
                border-color: #9B59B6 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-Models-get.is-open .opblock-summary {
                border-color: #9B59B6 !important;
            }

            /* Health endpoint (Green) */
            .swagger-ui #operations-Health-get .opblock-summary-method,
            .swagger-ui #operations-Health-get .btn,
            .swagger-ui #operations-Health-get .execute,
            .swagger-ui #operations-Health-get .try-out__btn {
                background: #2ECC71 !important;
                border-color: #2ECC71 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-Health-get.is-open .opblock-summary {
                border-color: #2ECC71 !important;
            }

            /* Predictions endpoint (Orange) */
            .swagger-ui #operations-Predictions-post .opblock-summary-method,
            .swagger-ui #operations-Predictions-post .btn,
            .swagger-ui #operations-Predictions-post .execute,
            .swagger-ui #operations-Predictions-post .try-out__btn {
                background: #E67E22 !important;
                border-color: #E67E22 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-Predictions-post.is-open .opblock-summary {
                border-color: #E67E22 !important;
            }

            /* Hide operation IDs */
            .swagger-ui .opblock-summary-operation-id {
                display: none !important;
            }

            /* Button hover effects */
            .swagger-ui #operations-Root-get .opblock-summary-method:hover,
            .swagger-ui #operations-Root-get .btn:hover {
                background: #FFE44D !important;
            }
            .swagger-ui #operations-Models-get .opblock-summary-method:hover,
            .swagger-ui #operations-Models-get .btn:hover {
                background: #A569BD !important;
            }
            .swagger-ui #operations-Health-get .opblock-summary-method:hover,
            .swagger-ui #operations-Health-get .btn:hover {
                background: #27AE60 !important;
            }
            .swagger-ui #operations-Predictions-post .opblock-summary-method:hover,
            .swagger-ui #operations-Predictions-post .btn:hover {
                background: #D35400 !important;
            }

            /* Active button styles */
            .swagger-ui .try-out__btn:active {
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.2) !important;
            }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = () => {
                const ui = SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    displayRequestDuration: true,
                    filter: false,
                    operationsSorter: 'alpha',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ]
                });

                // Additional styling for buttons after UI loads
                setTimeout(() => {
                    const applyColors = () => {
                        // Root endpoint (Yellow)
                        const rootElements = document.querySelectorAll('#operations-Root-get button');
                        rootElements.forEach(el => {
                            el.style.setProperty('background', '#FFD700', 'important');
                            el.style.setProperty('border-color', '#FFD700', 'important');
                            el.style.setProperty('color', '#000000', 'important');
                        });

                        // Models endpoint (Purple)
                        const modelsElements = document.querySelectorAll('#operations-Models-get button');
                        modelsElements.forEach(el => {
                            el.style.setProperty('background', '#9B59B6', 'important');
                            el.style.setProperty('border-color', '#9B59B6', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });

                        // Health endpoint (Green)
                        const healthElements = document.querySelectorAll('#operations-Health-get button');
                        healthElements.forEach(el => {
                            el.style.setProperty('background', '#2ECC71', 'important');
                            el.style.setProperty('border-color', '#2ECC71', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });

                        // Predictions endpoint (Orange)
                        const predictionElements = document.querySelectorAll('#operations-Predictions-post button');
                        predictionElements.forEach(el => {
                            el.style.setProperty('background', '#E67E22', 'important');
                            el.style.setProperty('border-color', '#E67E22', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });
                    };

                    // Apply colors initially
                    applyColors();

                    // Reapply colors when sections are expanded
                    const observer = new MutationObserver(applyColors);
                    observer.observe(document.getElementById('swagger-ui'), {
                        childList: true,
                        subtree: true
                    });
                }, 100);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)



# from fastapi import FastAPI, HTTPException, Path, Body
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import pandas as pd
# import joblib
# import os
# import sys
# import logging
# from datetime import datetime
# from dotenv import load_dotenv
# from fastapi.openapi.utils import get_openapi
# from sklearn.ensemble import VotingClassifier
# from sklearn.preprocessing import PolynomialFeatures
# from typing import Optional
# from contextlib import asynccontextmanager
# import socket
# from fastapi.responses import HTMLResponse

# # Logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('model_deployment.log'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# load_dotenv()
# MODEL_PATH = os.getenv("MODEL_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))
# logger.info(f"Using model path: {MODEL_PATH}")
# models = {}

# BEST_MODELS = {
#     "ml_olympiad_improved_final": "ML Olympiad – Improved XGBoost",
#     "archive_improved_final": "Archive – Improved Ensemble"
# }

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         logger.info(f"Starting model loading from {MODEL_PATH}")
#         if not os.path.exists(MODEL_PATH):
#             raise Exception(f"Model directory not found at {MODEL_PATH}")
#         model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
#         for model_file in model_files:
#             model_name = model_file.replace('.pkl', '')
#             if model_name in BEST_MODELS:
#                 model_path = os.path.join(MODEL_PATH, model_file)
#                 model_artifacts = joblib.load(model_path)
#                 models[model_name] = model_artifacts['model']
#         if not models:
#             raise Exception(f"No best models found for deployment in {MODEL_PATH}.")
#         logger.info("All models loaded successfully")
#     except Exception as e:
#         logger.error(f"Error during startup: {str(e)}")
#         raise e
#     yield
#     logger.info("Cleaning up models")
#     models.clear()

# app = FastAPI(
#     lifespan=lifespan,
#     title="Smoking Status Prediction API",
#     description="API for predicting smoking status using machine learning models",
#     version="2.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def get_ip():
#     try:
#         hostname = socket.gethostname()
#         local_ip = socket.gethostbyname(hostname)
#         return f"Hostname: {hostname}, Local IP: {local_ip}"
#     except Exception as e:
#         return f"Could not determine IP: {str(e)}"

# @app.on_event("startup")
# async def startup_event():
#     logger.info("=== Server Starting ===")
#     logger.info(get_ip())
#     logger.info("Access the API at: http://127.0.0.1:8000")
#     logger.info("Docs: http://127.0.0.1:8000/docs")

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Smoking Status Prediction API",
#         version="2.0.0",
#         description="**API for predicting smoking status using best-performing or ensemble ML models**",
#         routes=app.routes,
#     )
#     openapi_schema["tags"] = [
#         {"name": "Root", "description": "**Root endpoint operations**", "x-tag-style": {"background-color": "#FFD700"}},
#         {"name": "Models", "description": "**Model listing operations**", "x-tag-style": {"background-color": "#DC143C"}},
#         {"name": "Health", "description": "**Health check operations**", "x-tag-style": {"background-color": "#4CAF50"}},
#         {"name": "Predictions", "description": "**Smoking status prediction operations**", "x-tag-style": {"background-color": "#2196F3"}}
#     ]
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi

# class SmokingPredictionInput(BaseModel):
#     height_cm: float = Field(..., alias="height(cm)")
#     weight_kg: float = Field(..., alias="weight(kg)")
#     waist_cm: float = Field(..., alias="waist(cm)")
#     age: float
#     ALT: float
#     Gtp: float
#     HDL: float
#     LDL: float = Field(0.0)
#     Cholesterol: float = Field(0.0)
#     systolic: float
#     relaxation: float
#     hemoglobin: float
#     serum_creatinine: float = Field(..., alias="serum creatinine")
#     triglyceride: float
#     AST: Optional[float] = Field(0.0)
#     dental_caries: Optional[int] = Field(0, alias="dental caries")
#     eyesight_right: Optional[float] = Field(0.0, alias="eyesight(right)")
#     eyesight_left: Optional[float] = Field(0.0, alias="eyesight(left)")
#     fasting_blood_sugar: Optional[float] = Field(0.0, alias="fasting blood sugar")

#     class Config:
#         populate_by_name = True
#         json_schema_extra = {
#             "example": {
#                 "height(cm)": 170.0,
#                 "weight(kg)": 70.0,
#                 "waist(cm)": 85.0,
#                 "eyesight(left)": 1.0,
#                 "eyesight(right)": 1.0,
#                 "age": 35.0,
#                 "ALT": 25.0,
#                 "AST": 20.0,
#                 "Gtp": 30.0,
#                 "HDL": 50.0,
#                 "LDL": 100.0,
#                 "Cholesterol": 180.0,
#                 "dental caries": 0,
#                 "fasting blood sugar": 90.0,
#                 "relaxation": 80.0,
#                 "serum creatinine": 1.0,
#                 "triglyceride": 150.0,
#                 "hemoglobin": 15.0,
#                 "systolic": 120.0
#             }
#         }

# @app.get("/", tags=["Root"])
# async def root():
#     network_info = get_ip()
#     logger.info(f"Root endpoint accessed. {network_info}")
#     return {
#         "message": "Welcome to the Enhanced Smoking Prediction API",
#         "models_available": list(models.keys()),
#         "model_path": MODEL_PATH,
#         "status": "healthy",
#         "network_info": network_info,
#         "docs_url": "/docs"
#     }

# @app.get("/health", tags=["Health"])
# async def health_check():
#     logger.info("Health check endpoint accessed")
#     return {
#         "status": "healthy",
#         "models_loaded": list(models.keys()),
#         "model_path": MODEL_PATH,
#         "timestamp": datetime.now().isoformat()
#     }

# @app.get("/models", tags=["Models"])
# async def list_models():
#     logger.info("Models endpoint accessed")
#     return {
#         "available_models": BEST_MODELS,
#         "loaded_models": list(models.keys()),
#         "total": len(BEST_MODELS),
#         "model_path": MODEL_PATH
#     }
