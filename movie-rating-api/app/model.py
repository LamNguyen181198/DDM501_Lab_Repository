from surprise import SVD
import pickle
from pathlib import Path


class MovieRatingModel:
    def __init__(self, model_path: str = 'models/svd_model.pkl'):
        self.model = self._load_model(model_path)

    def _load_model(self, path: str):
        """Load a serialized SVD model from disk with error handling."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at {path}. "
                "Please ensure the model is trained and saved at the specified location."
            )
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"Error unpickling model from {path}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {path}: {str(e)}")

    def predict(self, user_id: str, movie_id: str) -> float:
        """Predict rating with input validation and error handling."""
        try:
            # Input validation
            if not user_id or not str(user_id).strip():
                raise ValueError("user_id cannot be empty")
            if not movie_id or not str(movie_id).strip():
                raise ValueError("movie_id cannot be empty")
            
            prediction = self.model.predict(str(user_id), str(movie_id))
            return round(prediction.est, 2)
        except ValueError as e:
            raise ValueError(f"Invalid input: {str(e)}")
        except Exception as e:
            raise RuntimeError(
                f"Error during prediction for user {user_id} and movie {movie_id}: {str(e)}"
            )

    def predict_batch(self, pairs: list) -> list:
        """Predict ratings for multiple pairs with error handling."""
        try:
            if not pairs:
                raise ValueError("Pairs list cannot be empty")
            return [self.predict(uid, mid) for uid, mid in pairs]
        except Exception as e:
            raise RuntimeError(f"Error in batch prediction: {str(e)}")