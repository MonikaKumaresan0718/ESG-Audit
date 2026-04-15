"""
Structured JSON logging for the ESG Auditor platform.
Provides consistent log formatting across all modules with
optional file output and log-level control via environment variables.
"""

import logging
import logging.config
import sys
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON log formatter that produces structured log records
    suitable for log aggregation systems (ELK, Loki, CloudWatch).
    """

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback
        from datetime import datetime, timezone

        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge any extra fields passed to the logger
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            } and not key.startswith("_"):
                log_entry[key] = value

        # Exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development environments.
    """

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"
        location = f"{record.name}.{record.funcName}:{record.lineno}"
        base = f"{level} | {location:<50} | {record.getMessage()}"

        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


def _build_logging_config(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Build logging.config dict."""

    formatter_class = (
        "core.logging.JSONFormatter"
        if log_format == "json"
        else "core.logging.TextFormatter"
    )

    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
            "level": log_level,
        }
    }

    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "formatter": "default",
            "level": log_level,
        }

    active_handlers = list(handlers.keys())

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": formatter_class,
            }
        },
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": active_handlers,
        },
        "loggers": {
            # Silence noisy third-party loggers
            "uvicorn": {"level": "WARNING", "propagate": True},
            "uvicorn.error": {"level": "WARNING", "propagate": True},
            "uvicorn.access": {"level": "WARNING", "propagate": True},
            "httpx": {"level": "WARNING", "propagate": True},
            "httpcore": {"level": "WARNING", "propagate": True},
            "transformers": {"level": "WARNING", "propagate": True},
            "sentence_transformers": {"level": "WARNING", "propagate": True},
            "faiss": {"level": "WARNING", "propagate": True},
            "celery": {"level": "INFO", "propagate": True},
            "sqlalchemy.engine": {
                "level": "WARNING",
                "propagate": True,
            },
            # Application loggers
            "agents": {"level": log_level, "propagate": True},
            "tools": {"level": log_level, "propagate": True},
            "ml": {"level": log_level, "propagate": True},
            "api": {"level": log_level, "propagate": True},
            "core": {"level": log_level, "propagate": True},
            "tasks": {"level": log_level, "propagate": True},
        },
    }


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.
    Called once at application startup.

    Args:
        log_level: Override log level (default: from settings).
        log_format: "json" or "text" (default: from settings).
        log_file: Optional log file path (default: from settings).
    """
    try:
        from core.config import settings as _settings
        level = log_level or _settings.LOG_LEVEL
        fmt = log_format or _settings.LOG_FORMAT
        fpath = log_file or _settings.LOG_FILE
    except Exception:
        level = "INFO"
        fmt = "text"
        fpath = None

    config = _build_logging_config(
        log_level=level,
        log_format=fmt,
        log_file=fpath,
    )
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Sets up basic logging if not already configured.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured Logger instance.
    """
    # Ensure at least basic config exists
    if not logging.root.handlers:
        try:
            setup_logging()
        except Exception:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                stream=sys.stdout,
            )

    return logging.getLogger(name)


# ── Audit-specific log helpers ───────────────────────────────────────────────

def log_audit_start(logger: logging.Logger, audit_id: str, company: str) -> None:
    """Log audit pipeline start with structured metadata."""
    logger.info(
        "ESG audit started",
        extra={"audit_id": audit_id, "company": company, "event": "audit_start"},
    )


def log_audit_end(
    logger: logging.Logger,
    audit_id: str,
    company: str,
    score: float,
    tier: str,
    duration: float,
) -> None:
    """Log audit pipeline completion with structured result metadata."""
    logger.info(
        "ESG audit completed",
        extra={
            "audit_id": audit_id,
            "company": company,
            "composite_score": score,
            "risk_tier": tier,
            "duration_seconds": round(duration, 2),
            "event": "audit_complete",
        },
    )


def log_stage(
    logger: logging.Logger,
    audit_id: str,
    stage: str,
    status: str = "started",
    **kwargs: Any,
) -> None:
    """Log pipeline stage transitions."""
    logger.info(
        f"Pipeline stage {status}: {stage}",
        extra={
            "audit_id": audit_id,
            "stage": stage,
            "stage_status": status,
            "event": "pipeline_stage",
            **kwargs,
        },
    )