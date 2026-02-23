from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data


def test_predict_no_model():
    payload = {
        "title": "Test Movie",
        "genre": "Action",
        "year": 2020,
        "budget": 1000000.0,
        "runtime": 120,
    }
    response = client.post("/predict", json=payload)
    # Without a saved model the endpoint returns 503
    assert response.status_code == 503


def test_predict_invalid_year():
    payload = {
        "title": "Old Movie",
        "genre": "Drama",
        "year": 1800,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_with_mock_model():
    mock_model = type("MockModel", (), {"predict": lambda self, X: [7.5]})()

    with patch("app.main._model.model", mock_model):
        payload = {
            "title": "Mock Movie",
            "genre": "Comedy",
            "year": 2023,
            "budget": 5000000.0,
            "runtime": 95,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Mock Movie"
        assert 0.0 <= data["predicted_rating"] <= 10.0
