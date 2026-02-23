# Movie Rating API

A REST API built with **FastAPI** that exposes a machine-learning model for predicting movie ratings.

## Project Structure

```
movie-rating-api/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI application
│   ├── model.py       # ML model loading & prediction
│   ├── schemas.py     # Pydantic models
│   └── config.py      # Configuration
├── models/            # Saved ML models
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Getting Started

### Local development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.  
Interactive docs are at `http://localhost:8000/docs`.

## Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/health`  | Health check and model status      |
| POST   | `/predict` | Predict rating for a given movie   |

## Running Tests

```bash
pytest tests/
```
