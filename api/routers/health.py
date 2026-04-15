"""
Health check router – GET /healthz
Returns application health status including DB, Redis, and model availability.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from api.schemas import HealthResponse
from core.config import settings
from core.database import get_db
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Record startup time for uptime calculation
_START_TIME = time.monotonic()


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the health status of the ESG Auditor API and its dependencies.",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Comprehensive health check endpoint.

    Checks:
    - API application status
    - Database connectivity
    - Redis/Celery broker connectivity
    - ML model load status
    """
    uptime = time.monotonic() - _START_TIME

    # Database check
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.warning(f"Health check DB failed: {e}")
        db_status = f"error: {str(e)[:100]}"

    # Redis / Celery check
    redis_status = "ok"
    try:
        from core.celery_app import celery_app
        inspector = celery_app.control.inspect(timeout=1.0)
        ping_result = inspector.ping()
        if ping_result is None:
            redis_status = "no_workers"
    except Exception as e:
        redis_status = f"error: {str(e)[:50]}"

    # ML model check
    model_loaded = False
    try:
        import os
        model_loaded = os.path.exists(settings.MODEL_PATH)
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database=db_status,
        redis=redis_status,
        model_loaded=model_loaded,
        uptime_seconds=round(uptime, 2),
    )


@router.get(
    "/readyz",
    summary="Readiness Check",
    description="Kubernetes readiness probe – returns 200 only when fully ready.",
)
async def readiness_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Readiness probe for Kubernetes deployments."""
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@router.get(
    "/livez",
    summary="Liveness Check",
    description="Kubernetes liveness probe – returns 200 if the process is alive.",
)
async def liveness_check() -> Dict[str, str]:
    """Simple liveness probe – always returns ok if process is running."""
    return {"alive": "true"}