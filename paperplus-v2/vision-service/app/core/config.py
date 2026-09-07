from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    shared_secret: str = "change-me"
    storage_root: str = "./files"

    target_width: int = 1240
    target_height: int = 1754

    # TODO(Phase 3 follow-up): point these at the real model files when ported from the old repo's services/.
    bubble_model_path: str = "./models/bubble_model_quantized.tflite"
    blur_model_path: str = "./models/blur_model_optimized.tflite"


settings = Settings()
