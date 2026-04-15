"""
Celery application factory for async ESG audit task execution.
Uses Redis as both broker and result backend.
"""

from celery import Celery
from celery.schedules import crontab
from celery.signals import task_prerun,task_postrun,task_failure
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


def create_celery_app() -> Celery:
    """
    Factory function that creates and configures the Celery application.

    Returns:
        Configured Celery instance.
    """
    app = Celery(
        "esg_auditor",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=[
            "tasks.audit_pipeline",
            "tasks.model_retrain",
        ],
    )

    app.conf.update(
        # Serialization
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        # Timezone
        timezone="UTC",
        enable_utc=True,
        # Task behavior
        task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
        task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Result backend
        result_expires=86400,  # 24 hours
        result_backend_transport_options={
            "retry_policy": {
                "timeout": 5.0,
            }
        },
        # Worker
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=50,
        worker_concurrency=2,
        # Retry
        task_default_retry_delay=30,
        task_max_retries=3,
        # Routing
        task_routes={
            "tasks.audit_pipeline.run_esg_audit_task": {
                "queue": "audit",
            },
            "tasks.model_retrain.retrain_model_task": {
                "queue": "maintenance",
            },
        },
        # Default queue
        task_default_queue="audit",
        task_queues={
            "audit": {"exchange": "audit", "routing_key": "audit"},
            "maintenance": {"exchange": "maintenance", "routing_key": "maintenance"},
        },
        # Beat schedule for periodic tasks
        beat_schedule={
            "retrain-model-weekly": {
                "task": "tasks.model_retrain.retrain_model_task",
                "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
                "kwargs": {"force": False},
            },
        },
    )

    # Set up task lifecycle hooks
    _setup_signals()

    logger.info(
        "Celery application initialized",
        extra={
            "broker": settings.CELERY_BROKER_URL,
            "backend": settings.CELERY_RESULT_BACKEND,
        },
    )

    return app


def _setup_signals() -> None:
    """Register Celery signal handlers for monitoring and logging."""

    @task_prerun.connect
    def task_prerun_handler(task_id, task, *args, **kwargs):
        logger.info(
            f"Task starting: {task.name}",
            extra={"celery_task_id": task_id, "task_name": task.name},
        )

    @task_postrun.connect
    def task_postrun_handler(task_id, task, retval, state, *args, **kwargs):
        logger.info(
            f"Task finished: {task.name} [{state}]",
            extra={
                "celery_task_id": task_id,
                "task_name": task.name,
                "state": state,
            },
        )

    @task_failure.connect
    def task_failure_handler(task_id, exception, traceback, einfo, *args, **kwargs):
        logger.error(
            f"Task failed: {exception}",
            extra={
                "celery_task_id": task_id,
                "error": str(exception),
            },
        )


# ── Module-level singleton ────────────────────────────────────────────────────

celery_app = create_celery_app()
