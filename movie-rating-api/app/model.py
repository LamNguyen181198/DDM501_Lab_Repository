from surprise import SVD
import pickle
from pathlib import Path

class MovieRatingModel:
    def __init__(self, model_path: str = 'models/svd_model.pkl'):
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def predict(self, user_id: str, movie_id: str) -> float:
        prediction = self.model.predict(user_id, movie_id)
        return round(prediction.est, 2)

    def predict_batch(self, pairs: list) -> list:
        return [self.predict(uid, mid) for uid, mid in pairs]