import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
	model_path: str = os.getenv("MODEL_PATH", "models/svd_model.pkl")
	model_version: str = os.getenv("MODEL_VERSION", "1.0.0")


settings = Settings()
