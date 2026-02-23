from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class PredictionRequest(BaseModel):
    user_id: str = Field(..., examples=["196"])
    movie_id: str = Field(..., examples=["242"])

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    user_id: str
    movie_id: str
    predicted_rating: float
    model_version: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool