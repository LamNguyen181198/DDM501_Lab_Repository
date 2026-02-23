import pytest 
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    response = client.post("/predict",
    json={"user_id": "196", "movie_id": "242"})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rating" in data
    assert 1.0 <= data["predicted_rating"] <= 5.0

def test_predict_invalid_input(): 
    response = client.post("/predict", json={})
    assert response.status_code == 422 # Validation error