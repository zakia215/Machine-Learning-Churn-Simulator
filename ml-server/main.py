from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import mlflow
import logging
from typing import List, Dict, Optional
from prometheus_fastapi_instrumentator import Instrumentator
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLflow Model Server", version="1.0.0")

Instrumentator().instrument(app).expose(app)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn_log_reg")

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Connected to MLflow at: {MLFLOW_TRACKING_URI}")
except Exception as e:
    logger.error(f"Failed to connect to MLflow: {str(e)}")

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    model_stage: str
    timestamp: str

model = None
model_info = {
    "version": "none",
    "stage": "none",
    "last_loaded": None,
    "name": MLFLOW_MODEL_NAME
}

def load_production_model():
    """Load the latest production model from MLflow"""
    global model, model_info
    try:
        client = mlflow.tracking.MlflowClient()
        
        filter_string = f"name='{model_info['name']}'"
        versions = client.search_model_versions(filter_string)
        # use tag to filter
        filtered_versions = [v for v in versions if v.tags.get("stage") == "Production"]
        
        if not filtered_versions:
            raise Exception("No production model found")
            
        latest_version = sorted(filtered_versions, key=lambda x: x.version, reverse=True)[0]
        
        # For rollback then, make the previous as the latest version
        model_uri = f"models:/{model_info['name']}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_info.update({
            "version": latest_version.version,
            "stage": "Production",
            "last_loaded": datetime.now().isoformat()
        })
        
        logger.info(f"Loaded model version {latest_version.version} from MLflow")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        load_production_model()
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_info": model_info
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make predictions using the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = np.array(request.features)
        
        predictions = model.predict(features)
        
        predictions = predictions.tolist()
        
        return PredictionResponse(
            predictions=predictions,
            model_version=model_info["version"],
            model_stage=model_info["stage"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-model")
async def refresh_model(background_tasks: BackgroundTasks):
    """Endpoint to refresh the model (load latest production version)"""
    try:
        background_tasks.add_task(load_production_model)
        return {"status": "Model refresh initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get current model information"""
    return model_info