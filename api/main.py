"""
FastAPI application factory for the ESG Auditor API.
Initializes middleware, routers, lifecycle events, and OpenAPI configuration.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routers import audit, health, reports
from core.config import settings
from core.database import create_tables
from core.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan context manager.
    Handles startup and shutdown tasks.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    setup_logging()
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION}",
        extra={"environment": settings.ENVIRONMENT},
    )

    # Create database tables
    try:
        await create_tables()
        logger.info("Database ready")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)

    # Pre-warm ML model (optional; avoids cold-start on first request)
    if settings.AUTO_TRAIN_MODEL:
        try:
            import os

            if not os.path.exists(settings.MODEL_PATH):
                logger.info("Pre-warming ML model (first-time training)...")
                from ml.train import train_pipeline

                train_pipeline(save=True)
                logger.info("ML model trained and ready")
            else:
                logger.info("ML model artifact found; skipping pre-warm")
        except Exception as e:
            logger.warning(f"ML pre-warm failed (non-fatal): {e}")

    logger.info("ESG Auditor API started successfully")

    yield  # ── Application runs here ─────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("ESG Auditor API shutting down")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="ESG Auditor API",
        description=(
            "## Autonomous Multi-Agent ESG Auditor\n\n"
            "Production-ready ESG auditing platform combining **ML risk scoring** "
            "with **zero-shot NLP analysis** in a multi-agent CrewAI pipeline.\n\n"
            "### Key Features\n"
            "- 🤖 Multi-agent orchestration via CrewAI\n"
            "- 📊 Hybrid ML + Zero-Shot NLP risk scoring\n"
            "- 📋 GRI, SASB, TCFD regulatory validation\n"
            "- 🔍 SHAP & LIME explainability\n"
            "- 📄 JSON, Markdown, and PDF report generation\n"
            "- ⚡ Async execution via Celery + Redis\n"
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        response.headers["X-Process-Time"] = f"{duration:.4f}s"
        return response

    # ── Prometheus metrics middleware ─────────────────────────────────────────
    if settings.ENABLE_METRICS:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                excluded_handlers=["/healthz", "/livez", "/readyz"],
            ).instrument(app).expose(app, endpoint="/metrics")
            logger.info("Prometheus metrics enabled at /metrics")
        except ImportError:
            logger.debug("prometheus-fastapi-instrumentator not installed; skipping metrics")

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "path": str(request.url.path)},
        )

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc) -> JSONResponse:
        logger.error(f"Unhandled server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)[:200]},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    # Health endpoints at root level (no prefix)
    app.include_router(health.router)

    # Versioned API endpoints
    app.include_router(audit.router, prefix=settings.API_PREFIX)
    app.include_router(reports.router, prefix=settings.API_PREFIX)

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "docs": "/docs",
                "health": "/healthz",
                "api": settings.API_PREFIX,
            }
        )

    logger.info(
        f"FastAPI app configured with {len(app.routes)} routes",
        extra={"prefix": settings.API_PREFIX},
    )

    return app


# ── Application instance ──────────────────────────────────────────────────────

app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1 if settings.DEBUG else 4,
    )