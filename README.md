# Movie Rating Prediction API

A FastAPI-based REST API for predicting movie ratings using a machine learning model trained with collaborative filtering (SVD algorithm).

## Project Features

- **Collaborative Filtering**: Uses Singular Value Decomposition (SVD) for accurate rating predictions
- **Fast API**: Built with FastAPI for high performance and automatic API documentation
- **Docker Support**: Containerized application for easy deployment
- **Comprehensive Testing**: Includes unit tests with code coverage reporting
- **Health Monitoring**: Built-in health check endpoint
- **Interactive API Documentation**: Auto-generated Swagger UI at `/docs`
- **Input Validation**: Pydantic-based request validation with detailed error messages

## Prerequisites

- **Python**: 3.10 or higher
- **Docker**: (Optional) For containerized deployment
- **Docker Compose**: (Optional) For orchestrating services

## Installation

### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LamNguyen181198/DDM501_Lab_Repository.git
   ```

2. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model file exists**:
   ```bash
   # The model file should be located at models/svd_model.pkl
   # If it doesn't exist, train it first using:
   python -m app.train_model
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be available at `http://localhost:8000`

### Option 2: Docker Installation

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

   The API will be available at `http://localhost:8000`

2. **Stop the service**:
   ```bash
   docker-compose down
   ```

## API Usage Examples

### Interactive API Documentation

Once the application is running, access the interactive Swagger UI:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Health Check Endpoint

Check if the API and model are loaded correctly:

**Request**:
```bash
curl -X GET "http://localhost:8000/health"
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Movie Rating

Predict a rating for a specific user-movie pair:

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "196",
    "movie_id": "242"
  }'
```

**Response**:
```json
{
  "user_id": "196",
  "movie_id": "242",
  "predicted_rating": 4.25,
  "model_version": "1.0.0"
}
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Tests with Coverage Report

```bash
pytest tests/ -v --cov=app --cov-report=html
```

This generates an HTML coverage report in the `htmlcov/` directory. Open `htmlcov/index.html` in your browser to view detailed coverage information.

### Run Specific Test File

```bash
pytest tests/test_api.py -v
```

### Test Output Example

```
tests/test_api.py::test_health_check PASSED                    [ 33%]
tests/test_api.py::test_predict_valid_input PASSED             [ 66%]
tests/test_api.py::test_predict_invalid_input PASSED           [100%]

====== 3 passed in 0.15s ======
```

## Project Structure

```
movie-rating-api/
├── app/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application and endpoints
│   ├── model.py                 # ML model loading and prediction
│   ├── schemas.py               # Pydantic request/response models
│   ├── config.py                # Application configuration
│   └── train_model.py           # Model training script
├── tests/
│   ├── __init__.py              # Test package initialization
│   ├── test_api.py              # API endpoint tests
│   └── __pycache__/             # Python cache directory
├── models/
│   └── svd_model.pkl            # Trained SVD model (pickle format)
├── htmlcov/                     # HTML code coverage reports
├── venv/                        # Python virtual environment
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Docker Compose service definition
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Directory Descriptions

- **app/**: Core application code
  - `main.py`: FastAPI application definition with all endpoints
  - `model.py`: MovieRatingModel class for loading and using the ML model
  - `schemas.py`: Pydantic models for request/response validation
  - `config.py`: Configuration management (environment variables, etc.)
  - `train_model.py`: Script to train the SVD model

- **tests/**: Test suite
  - `test_api.py`: Unit tests for API endpoints

- **models/**: Directory for storing trained models
  - `svd_model.pkl`: Serialized trained SVD model

- **htmlcov/**: Code coverage reports (generated after running tests with coverage)

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | Web framework for building APIs |
| uvicorn | 0.24.0 | ASGI server |
| pydantic | 2.5.2 | Data validation and settings |
| pandas | 2.1.3 | Data manipulation and analysis |
| numpy | 1.26.2 | Numerical computing |
| scikit-learn | 1.3.2 | Machine learning utilities |
| scikit-surprise | 1.1.3+ | Collaborative filtering algorithms |
| python-dotenv | 1.0.0 | Environment variable management |
| pytest | 7.4.3 | Testing framework |
| pytest-cov | 4.1.0 | Code coverage for pytest |
| httpx | 0.25.2 | HTTP client for testing |

## API Endpoints

### GET /health
- **Description**: Check API health and model status
- **Response**: HealthResponse object
- **Status Codes**: 200 (healthy)

### POST /predict
- **Description**: Predict a movie rating for a user
- **Request Body**: PredictionRequest (user_id, movie_id)
- **Response**: PredictionResponse (user_id, movie_id, predicted_rating, model_version)
- **Status Codes**: 200 (success), 422 (validation error), 500 (server error)

### POST /predict-batch
- **Description**: Predict ratings for multiple user-movie pairs
- **Request Body**: BatchPredictionRequest (array of user-movie pairs)
- **Response**: BatchPredictionResponse (array of predictions)
- **Status Codes**: 200 (success), 422 (validation error), 500 (server error)

## Error Handling

The API provides informative error messages:

- **422 Validation Error**: Invalid request body format
  ```json
  {
    "detail": [
      {
        "loc": ["body", "user_id"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
  ```

- **500 Server Error**: Internal server error during prediction
  ```json
  {
    "detail": "Error message describing the issue"
  }
  ```

## Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag automatically restarts the server when code changes are detected.

### Model Training

This will train the model on the MovieLens dataset and save it to `models/svd_model.pkl`.

## Production Deployment

For production deployment:

1. **Use a production ASGI server** (Gunicorn with Uvicorn workers):
   ```bash
   gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
   ```

2. **Use Docker** for containerization and orchestration

3. **Set appropriate environment variables** for configuration

4. **Enable HTTPS** using a reverse proxy (nginx, etc.)

5. **Monitor logs and metrics** using application monitoring tools

## Troubleshooting

### Model File Not Found
- **Error**: `FileNotFoundError: models/svd_model.pkl`
- **Solution**: Ensure the model file exists. Run `python -m app.train_model` to train it.

### Port Already in Use
- **Error**: `Address already in use`
- **Solution**: Use a different port: `uvicorn app.main:app --port 8001`

### Docker Build Issues
- **Error**: `failed to solve with frontend dockerfile.v0`
- **Solution**: Ensure Docker is running and has sufficient resources

## Model Information

- **Algorithm**: Singular Value Decomposition (SVD) from scikit-surprise
- **Type**: Collaborative Filtering
- **Input**: User ID and Movie ID
- **Output**: Predicted rating (1.0 - 5.0 scale)
- **Accuracy**: Depends on training data quality

## License

This project is part of the AI in MLOps coursework (DDM501).

## Support

For issues or questions, please refer to the project documentation or contact the course instructor.
