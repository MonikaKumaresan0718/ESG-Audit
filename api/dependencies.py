"""
api/dependencies.py
───────────────────
Reusable FastAPI dependency functions for the ESG Auditor API.

Provides:
  • get_db              – async SQLAlchemy session (per-request)
  • get_settings        – cached application settings
  • get_logger          – named logger for route handlers
  • verify_api_key      – optional bearer-token / x-api-key auth
  • rate_limit_check    – in-memory sliding-window rate limiter
  • pagination_params   – validated limit/offset query parameters
  • audit_exists        – 404-raising audit record resolver
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Annotated, Optional, Tuple

from fastapi import Depends, Header, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import Settings, get_settings
from core.database import AuditRecord, get_audit_record
from core.database import get_db as _get_db          # re-export from core
from core.logging import get_logger as _get_logger

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

_log = _get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Database session dependency
# ─────────────────────────────────────────────────────────────────────────────

async def get_db() -> AsyncSession:   # type: ignore[override]
    """
    FastAPI dependency – yields a per-request async SQLAlchemy session.

    Usage::

        @router.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            ...
    """
    async for session in _get_db():
        yield session


# Annotated alias for cleaner route signatures
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ─────────────────────────────────────────────────────────────────────────────
# Settings dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_app_settings() -> Settings:
    """
    FastAPI dependency – returns the cached application Settings instance.

    Usage::

        @router.get("/config")
        async def show_config(cfg: Settings = Depends(get_app_settings)):
            return {"version": cfg.APP_VERSION}
    """
    return get_settings()


AppSettings = Annotated[Settings, Depends(get_app_settings)]


# ─────────────────────────────────────────────────────────────────────────────
# Named logger dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_route_logger(request: Request):
    """
    FastAPI dependency – returns a logger scoped to the request path.

    Usage::

        @router.post("/audit")
        async def create_audit(logger = Depends(get_route_logger)):
            logger.info("Audit triggered")
    """
    return _get_logger(f"api.route.{request.url.path.strip('/').replace('/', '.')}")


RouteLogger = Annotated[object, Depends(get_route_logger)]


# ─────────────────────────────────────────────────────────────────────────────
# API-key authentication (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _constant_time_compare(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    ha = hashlib.sha256(a.encode()).digest()
    hb = hashlib.sha256(b.encode()).digest()
    return ha == hb


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_app_settings),
) -> Optional[str]:
    """
    FastAPI dependency – optional API-key authentication.

    Accepts the key via:
      • ``X-API-Key: <key>`` header
      • ``Authorization: Bearer <key>`` header

    If ``SECRET_KEY`` is set to the default placeholder value the check is
    **skipped** (development mode).  In production set a strong SECRET_KEY
    and callers must supply it.

    Returns:
        The validated key string, or ``None`` when auth is disabled.

    Raises:
        HTTPException 401: when a key is required but missing or invalid.
    """
    # Auth disabled in dev when key is the default placeholder
    default_key = "change-me-in-production-32chars!!"
    if settings.SECRET_KEY == default_key or not settings.SECRET_KEY:
        return None

    # Extract bearer token
    bearer_token: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()

    provided_key = x_api_key or bearer_token

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required.  Provide via X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _constant_time_compare(provided_key, settings.SECRET_KEY):
        _log.warning(
            "Invalid API key attempt",
            extra={"path": str(request.url.path), "ip": request.client.host if request.client else "unknown"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_key


# Annotated alias
AuthDep = Annotated[Optional[str], Depends(verify_api_key)]


# ─────────────────────────────────────────────────────────────────────────────
# In-memory sliding-window rate limiter
# ─────────────────────────────────────────────────────────────────────────────

# { client_key: [timestamp, ...] }
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_WINDOW = 60.0  # 1-minute sliding window


def rate_limit_check(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> None:
    """
    FastAPI dependency – enforces per-IP sliding-window rate limiting.

    Limit is set by ``API_RATE_LIMIT`` (requests per minute).
    When exceeded, raises HTTP 429.

    Usage::

        @router.post("/audit", dependencies=[Depends(rate_limit_check)])
        async def create_audit(...): ...
    """
    if settings.API_RATE_LIMIT <= 0:
        return  # rate limiting disabled

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - _WINDOW

    # Evict expired timestamps
    timestamps = _rate_limit_store[client_ip]
    _rate_limit_store[client_ip] = [ts for ts in timestamps if ts > window_start]

    if len(_rate_limit_store[client_ip]) >= settings.API_RATE_LIMIT:
        oldest = _rate_limit_store[client_ip][0]
        retry_after = int(_WINDOW - (now - oldest)) + 1
        _log.warning(
            "Rate limit exceeded",
            extra={"client_ip": client_ip, "limit": settings.API_RATE_LIMIT},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {settings.API_RATE_LIMIT} requests/min.  "
                f"Retry after {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    _rate_limit_store[client_ip].append(now)


# ─────────────────────────────────────────────────────────────────────────────
# Pagination query parameters
# ─────────────────────────────────────────────────────────────────────────────

class PaginationParams:
    """Reusable pagination query parameters (limit + offset)."""

    def __init__(
        self,
        limit: int = Query(default=20, ge=1, le=100, description="Number of records to return"),
        offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    ) -> None:
        self.limit = limit
        self.offset = offset

    def __repr__(self) -> str:
        return f"PaginationParams(limit={self.limit}, offset={self.offset})"


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]


# ─────────────────────────────────────────────────────────────────────────────
# Audit record resolver
# ─────────────────────────────────────────────────────────────────────────────

async def get_audit_or_404(
    audit_id: str,
    db: AsyncSession = Depends(get_db),
) -> AuditRecord:
    """
    FastAPI dependency – resolve an ``AuditRecord`` by ID or raise HTTP 404.

    Usage::

        @router.get("/{audit_id}")
        async def get_audit(record: AuditRecord = Depends(get_audit_or_404)):
            return record
    """
    record = await get_audit_record(db, audit_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit '{audit_id}' not found.",
        )
    return record


AuditDep = Annotated[AuditRecord, Depends(get_audit_or_404)]


# ─────────────────────────────────────────────────────────────────────────────
# Composite dependency: authenticated + rate-limited
# ─────────────────────────────────────────────────────────────────────────────

async def secure_endpoint(
    _auth: AuthDep,
    _rate: None = Depends(rate_limit_check),
) -> None:
    """
    Composite dependency that enforces both API-key auth and rate limiting.

    Usage::

        @router.post("/audit", dependencies=[Depends(secure_endpoint)])
        async def create_audit(...): ...
    """
    pass  # validation happens inside child dependencies


# ─────────────────────────────────────────────────────────────────────────────
# Request-ID middleware helper
# ─────────────────────────────────────────────────────────────────────────────

def get_request_id(request: Request) -> str:
    """
    FastAPI dependency – return the X-Request-ID header or a generated UUID.

    Usage::

        @router.get("/example")
        async def example(req_id: str = Depends(get_request_id)):
            logger.info("Handling request", extra={"request_id": req_id})
    """
    import uuid
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


RequestID = Annotated[str, Depends(get_request_id)]