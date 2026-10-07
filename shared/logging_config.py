"""Structured logging setup shared by api-service and vision-service (Phase 5).

Human-readable text format, but with correlation_id and service_name baked into every line via
a logging.Filter reading from a contextvar -- set once per request at each service's entrypoint
(api-service's webhook.py, per message; vision-service's main.py's /process handler), every log
call anywhere in that request's call stack then automatically carries it without threading
correlation_id through as a manual argument to each individual logger.info(...) call.

Without this, app.* loggers have no handler anywhere in their chain when run under real uvicorn:
uvicorn's own LOGGING_CONFIG only attaches handlers to its own uvicorn/uvicorn.access loggers,
never the root logger, so logger.info(...) calls are silently dropped and logger.error(...) only
shows via Python's bare last-resort fallback (no timestamp, no correlation_id). configure_logging
must be called once at process startup (module level in main.py is fine).
"""

import contextvars
import logging
import sys

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")


class _ContextFilter(logging.Filter):
    def __init__(self, service_name: str):
        super().__init__()
        self._service_name = service_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.service_name = self._service_name
        record.correlation_id = correlation_id_var.get()
        return True


def configure_logging(service_name: str, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(service_name)s] [correlation_id=%(correlation_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_ContextFilter(service_name))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
