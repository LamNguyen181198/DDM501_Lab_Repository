from pydantic import BaseModel, Field
from typing import Optional


class MovieFeatures(BaseModel):
    title: str = Field(..., description="Movie title")
    genre: str = Field(..., description="Movie genre")
    year: int = Field(..., ge=1888, le=2100, description="Release year")
    budget: Optional[float] = Field(None, ge=0, description="Budget in USD")
    runtime: Optional[int] = Field(None, ge=1, description="Runtime in minutes")


class RatingResponse(BaseModel):
    title: str
    predicted_rating: float = Field(..., ge=0.0, le=10.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
