import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent


def _path_is_within_repo(candidate: Path, repo_root: Path) -> bool:
    """Return True when a path is inside the current project checkout."""
    try:
        candidate.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def resolve_project_path(*parts: str) -> str:
    """Return project-relative paths that work from a local checkout or Colab clone."""
    override_root = os.getenv('PAPERPLUS_PROJECT_ROOT')
    base_dir = Path(override_root) if override_root else REPO_ROOT
    base_dir = base_dir.expanduser()
    if not base_dir.is_absolute():
        base_dir = (REPO_ROOT / base_dir).resolve()
    return str(base_dir.joinpath(*parts).resolve())


def resolve_config_path(env_name: str, *default_parts: str) -> str:
    """Prefer an explicit project-root override; otherwise use the repo-local config path when valid."""
    override_root = os.getenv('PAPERPLUS_PROJECT_ROOT')
    if override_root:
        return resolve_project_path(*default_parts)

    raw_value = os.getenv(env_name)
    if raw_value:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            return str((REPO_ROOT / candidate).resolve())
        if candidate.exists() and _path_is_within_repo(candidate, REPO_ROOT):
            return str(candidate)
    return resolve_project_path(*default_parts)


class Config:
    # Server
    SERVER_IP = os.getenv('SERVER_IP')

    # When true, outgoing WhatsApp/Exotel sends are logged instead of dispatched
    LOCAL_MODE = os.getenv('LOCAL_MODE', 'false').lower() == 'true'

    # External Services
    SHEETS_LOGGING_URL = os.getenv('SHEETS_LOGGING_URL')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Exotel / WhatsApp
    EXOTEL_SID = os.getenv('EXOTEL_SID')
    EXOTEL_KEY = os.getenv('EXOTEL_KEY')
    EXOTEL_TOKEN = os.getenv('EXOTEL_TOKEN')
    EXOTEL_SUBDOMAIN = os.getenv('EXOTEL_SUBDOMAIN')
    WHATSAPP_FROM = "+912071173227"

    # Paths
    DOWNLOADS_PATH = os.getenv('DOWNLOADS_PATH')
    DEWARPED_PATH = os.getenv('DEWARPED_PATH')
    DEBUG_PATH = os.getenv('DEBUG_PATH')
    CHECKED_PATH = os.getenv('CHECKED_PATH')
    LOGS_PATH = os.getenv('LOGS_PATH')

    # Image / OMR Settings
    TARGET_WIDTH = 1240
    TARGET_HEIGHT = 1754

    DEFAULT_NUM_QUESTIONS = 20
    NUM_ROW_TAGS = 10

    # roi format: (x_offset, y_offset, width, height)
    LEFT_QUESTION_ROI = (85, -40, 485, 90)
    RIGHT_QUESTION_ROI = (620, -40, 485, 90)

    ROLL_NUMBER_ROI = (420, 1660, 850, 1754) # x1, y1, x2, y2

    MIN_MARK_AREA = 600
    MAX_MARK_AREA = 950
    FILL_THRESHOLD = 0.6
    MIN_CIRCULARITY = 0.75

    # Worksheet levels map to difficulty level distributions
    # Keys are difficulty levels, values are proportions of 20 questions
    WORKSHEET_LEVEL_DISTRIBUTIONS = {
        "A": {1: 1.0},
        "B": {1: 0.5, 2: 0.5},
        "C": {1: 0.25, 2: 0.25, 3: 0.5},
        "D": {1: 0.125, 2: 0.125, 3: 0.25, 4: 0.5},  # 1-2 (25%) split evenly
        "E": {1: 1/12, 2: 1/12, 3: 1/12, 4: 0.25, 5: 0.5},  # 1-3 (25%) split evenly
        "F": {1: 1/16, 2: 1/16, 3: 1/16, 4: 1/16, 5: 0.25, 6: 0.5},  # 1-4 (25%) split evenly
        "G": {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.25, 7: 0.5},  # 1-5 (25%) split evenly
    }

    # PDF Generation
    ORIENTATION_ID = 586
    TAGS_PATH = resolve_config_path('TAGS_PATH', 'assets', 'tags')
    TEMPLATES_PATH = resolve_config_path('TEMPLATES_PATH', 'assets', 'templates')
    PDF_WRITE_PATH = resolve_config_path('PDF_WRITE_PATH', 'files', 'pdf')
    WORKSHEET_JSON_PATH = resolve_config_path('WORKSHEET_JSON_PATH', 'files', 'json')
    HTML_BASE_DIR = resolve_config_path('HTML_BASE_DIR', 'assets')

    # Database
    DATABASE_URL = os.getenv('DATABASE_URL')

    HOWTO_IMAGE_URL = os.getenv('HOWTO_IMAGE_URL')

SETTINGS = Config()