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

    # Admin dashboard alert thresholds (app/services/monitoring.py). Env-overridable, e.g.
    # ALERT_FAILURE_RATE=0.5. Rules that need a minimum sample size stay quiet below it, so one
    # bad scan out of two doesn't flash a red banner.
    alert_min_scans_for_failure_rate: int = 5
    alert_failure_rate: float = 0.3  # failed / all scans over the last 24h
    alert_vision_failures_1h: int = 3  # scans lost to vision-service errors in the last hour
    alert_open_reviews: int = 10  # failed scans waiting for someone to review them
    alert_review_age_hours: int = 24  # oldest unreviewed failed scan
    alert_disk_warn_pct: float = 85.0
    alert_disk_critical_pct: float = 95.0
    alert_silence_hours: int = 48  # no scans at all for this long -> check the Exotel webhook; 0 disables


settings = Settings()
