import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model import MovieRatingModel

client = TestClient(app)


class TestMovieRatingModel:
    """Test suite for MovieRatingModel class."""
    
    def test_model_initialization(self):
        """Test model initializes successfully."""
        model = MovieRatingModel()
        assert model.model is not None
    
    def test_model_predict_returns_float(self):
        """Test predict returns a float."""
        model = MovieRatingModel()
        result = model.predict("196", "242")
        assert isinstance(result, float)
    
    def test_model_predict_in_valid_range(self):
        """Test predict returns rating in valid range."""
        model = MovieRatingModel()
        result = model.predict("196", "242")
        assert 0.0 <= result <= 5.0
    
    def test_model_predict_batch(self):
        """Test predict_batch returns list of floats."""
        model = MovieRatingModel()
        pairs = [("196", "242"), ("1", "1"), ("500", "300")]
        results = model.predict_batch(pairs)
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, float)
            assert 0.0 <= result <= 5.0
    
    def test_model_predict_batch_empty_list(self):
        """Test predict_batch raises error for empty list."""
        model = MovieRatingModel()
        with pytest.raises(RuntimeError):
            model.predict_batch([])
    
    def test_model_predict_with_different_ids(self):
        """Test predict with various user and movie IDs."""
        model = MovieRatingModel()
        test_cases = [
            ("196", "242"), ("1", "1"), ("100", "50"),
            ("999", "888"), ("5", "20")
        ]
        for user_id, movie_id in test_cases:
            result = model.predict(user_id, movie_id)
            assert isinstance(result, float)
            assert 0.0 <= result <= 5.0
    
    def test_model_predict_consistency(self):
        """Test predict returns same result for same inputs."""
        model = MovieRatingModel()
        result1 = model.predict("196", "242")
        result2 = model.predict("196", "242")
        assert result1 == result2
    
    def test_model_predict_empty_user_id_error(self):
        """Test predict handles empty user_id."""
        model = MovieRatingModel()
        with pytest.raises(ValueError):
            model.predict("", "242")
    
    def test_model_predict_empty_movie_id_error(self):
        """Test predict handles empty movie_id."""
        model = MovieRatingModel()
        with pytest.raises(ValueError):
            model.predict("196", "")
    
    def test_model_file_not_found(self):
        """Test model initialization fails with non-existent file."""
        with pytest.raises(FileNotFoundError):
            MovieRatingModel("nonexistent/path/model.pkl")


class TestHealth:
    """Test suite for health check endpoint."""
    
    def test_health_check_success(self):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_health_check_response_structure(self):
        """Test health check response has correct structure."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert len(data) == 2  # Only 2 fields


class TestPredict:
    """Test suite for predict endpoint."""
    
    def test_predict_valid_input(self):
        """Test predict with valid input."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rating" in data
        assert "user_id" in data
        assert "movie_id" in data
        assert "model_version" in data
        assert data["user_id"] == "196"
        assert data["movie_id"] == "242"
    
    def test_predict_rating_in_valid_range(self):
        """Test predicted rating is within valid range (1.0-5.0)."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        data = response.json()
        assert 0.0 <= data["predicted_rating"] <= 5.0
    
    def test_predict_rating_is_float(self):
        """Test predicted rating is a float/numeric value."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        data = response.json()
        assert isinstance(data["predicted_rating"], (int, float))
    
    def test_predict_model_version_present(self):
        """Test model version is included in response."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        data = response.json()
        assert data["model_version"] == "1.0.0"
    
    def test_predict_different_user_movie_pair(self):
        """Test predict with different user-movie pair."""
        response = client.post("/predict", json={"user_id": "1", "movie_id": "1"})
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "1"
        assert data["movie_id"] == "1"
        assert 0.0 <= data["predicted_rating"] <= 5.0
    
    def test_predict_with_string_ids(self):
        """Test predict accepts string IDs."""
        response = client.post("/predict", json={"user_id": "500", "movie_id": "300"})
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "500"
        assert data["movie_id"] == "300"


class TestPredictValidation:
    """Test suite for input validation."""
    
    def test_predict_missing_user_id(self):
        """Test predict rejects missing user_id."""
        response = client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422
    
    def test_predict_missing_movie_id(self):
        """Test predict rejects missing movie_id."""
        response = client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422
    
    def test_predict_empty_body(self):
        """Test predict rejects empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_null_user_id(self):
        """Test predict rejects null user_id."""
        response = client.post("/predict", json={"user_id": None, "movie_id": "242"})
        assert response.status_code == 422
    
    def test_predict_null_movie_id(self):
        """Test predict rejects null movie_id."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": None})
        assert response.status_code == 422
    
    def test_predict_empty_string_user_id(self):
        """Test predict rejects empty string user_id."""
        response = client.post("/predict", json={"user_id": "", "movie_id": "242"})
        assert response.status_code == 422
    
    def test_predict_empty_string_movie_id(self):
        """Test predict rejects empty string movie_id."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": ""})
        assert response.status_code == 422
    
    def test_predict_invalid_json(self):
        """Test predict rejects invalid JSON."""
        response = client.post("/predict", content="invalid json", 
                              headers={"Content-Type": "application/json"})
        assert response.status_code == 422


class TestPredictErrorHandling:
    """Test suite for error handling."""
    
    def test_predict_error_response_format(self):
        """Test error response has proper format."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_predict_missing_field_error_detail(self):
        """Test error detail contains information about missing field."""
        response = client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        # Detail should be a list with error information
        assert isinstance(data["detail"], list)


class TestPredictResponseStructure:
    """Test suite for response structure validation."""
    
    def test_predict_response_has_all_fields(self):
        """Test response contains all required fields."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        data = response.json()
        required_fields = ["user_id", "movie_id", "predicted_rating", "model_version"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_predict_response_field_types(self):
        """Test response fields have correct types."""
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        data = response.json()
        assert isinstance(data["user_id"], str)
        assert isinstance(data["movie_id"], str)
        assert isinstance(data["predicted_rating"], (int, float))
        assert isinstance(data["model_version"], str)
    
    def test_predict_returns_input_ids(self):
        """Test response returns the same user_id and movie_id from request."""
        user_id = "999"
        movie_id = "888"
        response = client.post("/predict", json={"user_id": user_id, "movie_id": movie_id})
        data = response.json()
        assert data["user_id"] == user_id
        assert data["movie_id"] == movie_id


class TestEndpointAvailability:
    """Test suite for endpoint availability."""
    
    def test_health_endpoint_exists(self):
        """Test /health endpoint is available."""
        response = client.get("/health")
        assert response.status_code in [200, 405, 404]  # Should not be completely broken
        assert response.status_code != 404  # Should exist
    
    def test_predict_endpoint_exists(self):
        """Test /predict endpoint is available."""
        response = client.post("/predict", json={"user_id": "1", "movie_id": "1"})
        assert response.status_code in [200, 422]  # Should exist (200 or validation error 422)