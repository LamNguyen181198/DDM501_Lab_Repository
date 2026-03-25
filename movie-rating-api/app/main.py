from typing import Optional

from fastapi import FastAPI, HTTPException
from app.model import MovieRatingModel
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from app.config import settings

app = FastAPI(
    title="Movie Rating Prediction API",
    description="API for predicting movie ratings using collaborative filtering",
    version=settings.model_version,
)

model: Optional[MovieRatingModel] = None


def _load_model_safe() -> Optional[MovieRatingModel]:
    """Load model safely and return None if loading fails."""
    try:
        return MovieRatingModel(settings.model_path)
    except Exception:
        return None


model = _load_model_safe()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Information about model version, metrics"""
    return {
        "version": settings.model_version,
        "description": "Collaborative Filtering Model using SVD",
        "metrics": {
            "rmse": 0.85,
            "mae": 0.65
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict movie rating for a user using collaborative filtering (SVD).
    
    Args:
        user_id: Unique user identifier
        movie_id: Unique movie identifier
        
    Returns:
        Predicted rating (1.0-5.0 scale) with metadata
    """
    global model
    if model is None:
        model = _load_model_safe()

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        rating = model.predict(request.user_id, request.movie_id)
        return {
            "user_id": request.user_id,
            "movie_id": request.movie_id,
            "predicted_rating": rating,
            "model_version": settings.model_version,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")