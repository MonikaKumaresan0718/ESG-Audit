"""
Celery task: run_esg_audit_task
Executes the full ESG audit pipeline asynchronously and persists results to the DB.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from core.celery_app import celery_app
from core.logging import get_logger

logger = get_logger(__name__)


class AuditTask(Task):
    """
    Custom Celery Task base class with DB session handling
    and retry logic for the ESG audit pipeline.
    """

    abstract = True
    _db_session = None

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure by updating audit record in DB."""
        audit_id = kwargs.get("audit_id", "unknown")
        logger.error(
            f"Audit task failed permanently",
            extra={
                "audit_id": audit_id,
                "celery_task_id": task_id,
                "error": str(exc),
            },
        )
        self._update_db_sync(
            audit_id=audit_id,
            status="failed",
            error_message=str(exc)[:1000],
            completed_at=datetime.utcnow(),
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log task retry events."""
        audit_id = kwargs.get("audit_id", "unknown")
        logger.warning(
            f"Audit task retrying",
            extra={
                "audit_id": audit_id,
                "celery_task_id": task_id,
                "retry_reason": str(exc),
            },
        )

    def _update_db_sync(self, audit_id: str, **kwargs) -> None:
        """Synchronous DB update using a fresh event loop."""
        try:
            asyncio.run(self._async_update_db(audit_id, **kwargs))
        except Exception as e:
            logger.error(f"DB update failed for audit {audit_id}: {e}")

    @staticmethod
    async def _async_update_db(audit_id: str, **kwargs) -> None:
        """Async helper to update audit record."""
        from core.database import AsyncSessionLocal, update_audit_record

        async with AsyncSessionLocal() as db:
            await update_audit_record(db, audit_id, **kwargs)
            await db.commit()


@celery_app.task(
    bind=True,
    base=AuditTask,
    name="tasks.audit_pipeline.run_esg_audit_task",
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=600,
    time_limit=900,
    acks_late=True,
)
def run_esg_audit_task(
    self: AuditTask,
    audit_id: str,
    company_name: str,
    esg_data: Optional[Dict[str, Any]] = None,
    pdf_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    fetch_news: bool = False,
    ml_weight: float = 0.6,
    nlp_weight: float = 0.4,
) -> Dict[str, Any]:
    """
    Celery task: Execute the full ESG audit pipeline.

    Args:
        audit_id: Unique identifier for this audit run.
        company_name: Company to audit.
        esg_data: Optional structured ESG metrics dict.
        pdf_path: Optional path to sustainability report PDF.
        csv_path: Optional path to ESG metrics CSV.
        fetch_news: Whether to fetch news articles for NLP analysis.
        ml_weight: ML model weight in hybrid fusion (0–1).
        nlp_weight: NLP model weight in hybrid fusion (0–1).

    Returns:
        Full pipeline result dictionary.
    """
    logger.info(
        "Starting ESG audit task",
        extra={
            "audit_id": audit_id,
            "company": company_name,
            "celery_task_id": self.request.id,
        },
    )

    # Mark as running in DB
    self._update_db_sync(
        audit_id=audit_id,
        status="running",
        started_at=datetime.utcnow(),
    )

    try:
        # ── Execute pipeline ─────────────────────────────────────────────────
        from agents.orchestrator import OrchestratorAgent
        from agents.hybrid_fusion import HybridFusionAgent

        orchestrator = OrchestratorAgent()

        # Apply custom fusion weights if non-default
        if abs(ml_weight - 0.6) > 0.01 or abs(nlp_weight - 0.4) > 0.01:
            _original_stage = orchestrator._stage_hybrid_fusion

            def _custom_fusion_stage(ctx):
                fusion_agent = HybridFusionAgent(
                    ml_weight=ml_weight, nlp_weight=nlp_weight
                )
                fusion_result = fusion_agent.fuse(
                    ml_result=ctx.get("ml_risk", {}),
                    zero_shot_result=ctx.get("zero_shot", {}),
                )
                ctx["fusion"] = fusion_result
                ctx["pipeline_stages"].append("hybrid_fusion")
                return ctx

            orchestrator._stage_hybrid_fusion = _custom_fusion_stage

        result = orchestrator.run_audit_pipeline(
            company_name=company_name,
            esg_data=esg_data,
            pdf_path=pdf_path,
            csv_path=csv_path,
        )

        # ── Extract key metrics ───────────────────────────────────────────────
        fusion = result.get("fusion", {})
        ml_risk = result.get("ml_risk", {})
        validation = result.get("validation", {})
        report = result.get("report", {})

        composite_score = fusion.get("composite_esg_score")
        risk_tier = fusion.get("risk_tier")
        ml_score = ml_risk.get("risk_score_ml")
        val_status = validation.get("validation_status")

        # ── Update DB with results ────────────────────────────────────────────
        self._update_db_sync(
            audit_id=audit_id,
            status="completed",
            completed_at=datetime.utcnow(),
            composite_score=composite_score,
            risk_tier=risk_tier,
            ml_risk_score=ml_score,
            validation_status=val_status,
            result_json=result,
            json_report_path=report.get("json_report_path"),
            markdown_report_path=report.get("markdown_report_path"),
            pdf_report_path=report.get("pdf_report_path"),
        )

        logger.info(
            "ESG audit task completed",
            extra={
                "audit_id": audit_id,
                "composite_score": composite_score,
                "risk_tier": risk_tier,
                "celery_task_id": self.request.id,
            },
        )

        return {
            "audit_id": audit_id,
            "status": "completed",
            "composite_score": composite_score,
            "risk_tier": risk_tier,
        }

    except SoftTimeLimitExceeded:
        logger.error(
            "Audit task soft time limit exceeded",
            extra={"audit_id": audit_id},
        )
        self._update_db_sync(
            audit_id=audit_id,
            status="failed",
            completed_at=datetime.utcnow(),
            error_message="Task exceeded time limit (600s)",
        )
        raise

    except Exception as exc:
        logger.error(
            f"Audit task error: {exc}",
            extra={"audit_id": audit_id},
            exc_info=True,
        )
        # Retry on transient errors
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            self._update_db_sync(
                audit_id=audit_id,
                status="failed",
                completed_at=datetime.utcnow(),
                error_message=str(exc)[:1000],
            )
            raise