from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    shared_secret: str = "change-me"

    target_width: int = 1240
    target_height: int = 1754

    # Model files ship in vision-service/models/. The blur model is the one actually used (see bubble_inference.py).
    bubble_model_path: str = "./models/bubble_model_quantized.tflite"
    blur_model_path: str = "./models/blur_model_optimized.tflite"


settings = Settings()
