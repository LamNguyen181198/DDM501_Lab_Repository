from fastapi import FastAPI, HTTPException

from .config import settings
from .model import MovieRatingModel
from .schemas import HealthResponse, MovieFeatures, RatingResponse

app = FastAPI(title=settings.app_name, version=settings.app_version)

_model = MovieRatingModel(settings.model_path)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        model_loaded=_model.is_loaded,
    )


@app.post("/predict", response_model=RatingResponse)
def predict_rating(movie: MovieFeatures):
    if not _model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    rating, confidence = _model.predict(
        genre=movie.genre,
        year=movie.year,
        budget=movie.budget,
        runtime=movie.runtime,
    )
    return RatingResponse(
        title=movie.title,
        predicted_rating=rating,
        confidence=confidence,
    )
