from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "Movie Rating API"
    app_version: str = "1.0.0"
    model_path: str = "models/movie_rating_model.pkl"
    debug: bool = False


settings = Settings()
