from fastapi import FastAPI, HTTPException
from app.model import MovieRatingModel
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse

app = FastAPI( title="Movie Rating Prediction API", description="API for predicting movie ratings using collaborative filtering", version="1.0.0" )

# Load model at startup 
model = MovieRatingModel()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        rating = model.predict(request.user_id, request.movie_id)
        return {
            "user_id": request.user_id,
            "movie_id": request.movie_id,
            "predicted_rating": rating,
            "model_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))