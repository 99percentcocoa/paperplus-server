from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config; every value is overridable via environment variables (no hardcoded paths/secrets)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/paperplus_v2"

    vision_service_url: str = "http://localhost:8100"
    vision_service_shared_secret: str = "change-me"

    storage_root: str = "./files"
    # Base URL this service is publicly reachable at, used to build served URLs (checked-image
    # link sent over WhatsApp via Exotel, and the fileURL/checkedURL fields logged to Sheets) --
    # both require an internet-fetchable URL, not a local filesystem path.
    public_base_url: str = "http://localhost:8000"

    local_mode: bool = False  # when true, outgoing WhatsApp sends are logged instead of dispatched
    whatsapp_from: str = "+912071173227"
    howto_image_url: str = ""

    exotel_sid: str = ""
    exotel_key: str = ""
    exotel_token: str = ""
    exotel_subdomain: str = ""

    sheets_logging_url: str = ""


settings = Settings()
