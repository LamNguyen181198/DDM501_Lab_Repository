import os
import logging
from typing import Optional, Tuple

import joblib

logger = logging.getLogger(__name__)


class MovieRatingModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info("Model loaded from %s", self.model_path)
            except Exception as exc:
                logger.error("Failed to load model: %s", exc)
                self.model = None
        else:
            logger.warning("Model file not found at %s", self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, genre: str, year: int, budget: Optional[float], runtime: Optional[int]) -> Tuple[float, Optional[float]]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        # Encode genre as a numeric value for the model feature vector
        genre_encoded = abs(hash(genre)) % 1000
        features = [genre_encoded, year, budget or 0.0, runtime or 0]
        prediction = self.model.predict([features])[0]
        confidence = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([features])[0]
            confidence = float(max(proba))

        return float(prediction), confidence
