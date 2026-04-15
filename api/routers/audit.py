"""
Audit router – POST /v1/audit and GET /v1/audit/{audit_id}.

Supports both synchronous (inline) and asynchronous (Celery) audit execution.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    AuditCreateResponse,
    AuditRequest,
    AuditResult,
    AuditStatusEnum,
    AuditSummary,
    EmergingRisk,
    ErrorResponse,
    PaginatedAuditList,
    RiskSignal,
    ValidationFlag,
)
from core.database import (
    AuditRecord,
    AuditStatus,
    create_audit_record,
    get_audit_record,
    get_db,
    list_audit_records,
    update_audit_record,
)
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/audit", tags=["Audit"])


# ── POST /v1/audit ────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=AuditCreateResponse,
    status_code=202,
    summary="Trigger ESG Audit",
    description=(
        "Trigger a full ESG audit for a company. "
        "Returns immediately with an audit_id for async polling, "
        "or the full result if async_execution=False."
    ),
    responses={
        202: {"description": "Audit accepted and queued"},
        200: {"description": "Audit completed synchronously"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def create_audit(
    request: AuditRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AuditCreateResponse:
    """
    POST /v1/audit

    Accepts an AuditRequest and either:
    - Dispatches to Celery (async_execution=True, default), or
    - Runs inline and returns the full result immediately.
    """
    audit_id = str(uuid.uuid4())
    logger.info(
        "Audit request received",
        extra={"audit_id": audit_id, "company": request.company_name},
    )

    # Prepare ESG data dict
    esg_data_dict: Optional[Dict[str, Any]] = None
    if request.esg_data:
        esg_data_dict = {
            k: v
            for k, v in request.esg_data.dict().items()
            if v is not None
        }

    if request.async_execution:
        # ── Async path: submit to Celery ─────────────────────────────────────
        celery_task_id = None
        try:
            from tasks.audit_pipeline import run_esg_audit_task

            task = run_esg_audit_task.apply_async(
                kwargs={
                    "audit_id": audit_id,
                    "company_name": request.company_name,
                    "esg_data": esg_data_dict,
                    "pdf_path": request.pdf_path,
                    "csv_path": request.csv_path,
                    "fetch_news": request.fetch_news,
                    "ml_weight": request.ml_weight,
                    "nlp_weight": request.nlp_weight,
                },
                task_id=f"audit-{audit_id}",
            )
            celery_task_id = task.id
            logger.info(
                "Audit dispatched to Celery",
                extra={"audit_id": audit_id, "celery_task_id": celery_task_id},
            )
        except Exception as celery_err:
            logger.warning(
                f"Celery unavailable ({celery_err}); falling back to background task"
            )
            background_tasks.add_task(
                _run_audit_background,
                audit_id=audit_id,
                company_name=request.company_name,
                esg_data=esg_data_dict,
                pdf_path=request.pdf_path,
                csv_path=request.csv_path,
                fetch_news=request.fetch_news,
                ml_weight=request.ml_weight,
                nlp_weight=request.nlp_weight,
            )

        # Persist to DB
        await create_audit_record(db, audit_id, request.company_name, celery_task_id)

        return AuditCreateResponse(
            audit_id=audit_id,
            status=AuditStatusEnum.PENDING,
            message="Audit queued successfully. Poll the audit endpoint for results.",
            poll_url=f"/v1/audit/{audit_id}",
        )

    else:
        # ── Synchronous path: run inline ──────────────────────────────────────
        await create_audit_record(db, audit_id, request.company_name)
        await update_audit_record(
            db,
            audit_id,
            status=AuditStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        try:
            result = _execute_audit_pipeline(
                audit_id=audit_id,
                company_name=request.company_name,
                esg_data=esg_data_dict,
                pdf_path=request.pdf_path,
                csv_path=request.csv_path,
                fetch_news=request.fetch_news,
                ml_weight=request.ml_weight,
                nlp_weight=request.nlp_weight,
            )

            fusion = result.get("fusion", {})
            ml_risk = result.get("ml_risk", {})
            validation = result.get("validation", {})
            report = result.get("report", {})

            await update_audit_record(
                db,
                audit_id,
                status=AuditStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                composite_score=fusion.get("composite_esg_score"),
                risk_tier=fusion.get("risk_tier"),
                ml_risk_score=ml_risk.get("risk_score_ml"),
                validation_status=validation.get("validation_status"),
                result_json=result,
                json_report_path=report.get("json_report_path"),
                markdown_report_path=report.get("markdown_report_path"),
            )

            audit_result = _build_audit_result(audit_id, request.company_name, result, db)

            return AuditCreateResponse(
                audit_id=audit_id,
                status=AuditStatusEnum.COMPLETED,
                message="Audit completed successfully.",
                result=audit_result,
            )

        except Exception as exc:
            logger.error(
                "Synchronous audit failed",
                extra={"audit_id": audit_id, "error": str(exc)},
                exc_info=True,
            )
            await update_audit_record(
                db,
                audit_id,
                status=AuditStatus.FAILED,
                completed_at=datetime.utcnow(),
                error_message=str(exc)[:1000],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Audit pipeline failed: {str(exc)[:200]}",
            )


# ── GET /v1/audit/{audit_id} ──────────────────────────────────────────────────

@router.get(
    "/{audit_id}",
    response_model=AuditResult,
    summary="Get Audit Result",
    description="Retrieve the status and full results of an ESG audit by ID.",
    responses={
        200: {"description": "Audit result"},
        404: {"model": ErrorResponse, "description": "Audit not found"},
    },
)
async def get_audit(
    audit_id: str,
    db: AsyncSession = Depends(get_db),
) -> AuditResult:
    """GET /v1/audit/{audit_id}"""
    record = await get_audit_record(db, audit_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Audit '{audit_id}' not found",
        )

    # If completed, build full result from stored JSON
    if record.status == AuditStatus.COMPLETED and record.result_json:
        return _build_audit_result(
            audit_id=audit_id,
            company_name=record.company_name,
            pipeline_result=record.result_json,
            record=record,
        )

    # Otherwise return status-only response
    return AuditResult(
        audit_id=audit_id,
        company_name=record.company_name,
        status=AuditStatusEnum(record.status.value),
        created_at=record.created_at,
        completed_at=record.completed_at,
        duration_seconds=record.duration_seconds,
        composite_esg_score=record.composite_score,
        risk_tier=record.risk_tier,
        ml_risk_score=record.ml_risk_score,
        validation_status=record.validation_status,
        error_message=record.error_message,
    )


# ── GET /v1/audit ─────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=PaginatedAuditList,
    summary="List Audits",
    description="Retrieve a paginated list of all ESG audits.",
)
async def list_audits(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    company: Optional[str] = Query(default=None, description="Filter by company name"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedAuditList:
    """GET /v1/audit"""
    records = await list_audit_records(
        db, limit=limit + 1, offset=offset, company_filter=company
    )

    has_more = len(records) > limit
    items = records[:limit]

    summaries = [
        AuditSummary(
            audit_id=r.id,
            company_name=r.company_name,
            status=AuditStatusEnum(r.status.value),
            composite_score=r.composite_score,
            risk_tier=r.risk_tier,
            created_at=r.created_at,
            completed_at=r.completed_at,
            duration_seconds=r.duration_seconds,
        )
        for r in items
    ]

    return PaginatedAuditList(
        items=summaries,
        total=len(summaries),
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _execute_audit_pipeline(
    audit_id: str,
    company_name: str,
    esg_data: Optional[Dict[str, Any]],
    pdf_path: Optional[str],
    csv_path: Optional[str],
    fetch_news: bool,
    ml_weight: float,
    nlp_weight: float,
) -> Dict[str, Any]:
    """Run the full ESG audit pipeline synchronously."""
    from agents.orchestrator import OrchestratorAgent
    from agents.hybrid_fusion import HybridFusionAgent

    # Override fusion weights if custom
    orchestrator = OrchestratorAgent()

    # Monkey-patch fusion weights if non-default
    if abs(ml_weight - 0.6) > 0.01 or abs(nlp_weight - 0.4) > 0.01:
        original_fusion = orchestrator._stage_hybrid_fusion

        def custom_fusion(ctx):
            from agents.hybrid_fusion import HybridFusionAgent as HFA
            agent = HFA(ml_weight=ml_weight, nlp_weight=nlp_weight)
            fusion_result = agent.fuse(
                ml_result=ctx.get("ml_risk", {}),
                zero_shot_result=ctx.get("zero_shot", {}),
            )
            ctx["fusion"] = fusion_result
            ctx["pipeline_stages"].append("hybrid_fusion")
            return ctx

        orchestrator._stage_hybrid_fusion = custom_fusion

    return orchestrator.run_audit_pipeline(
        company_name=company_name,
        esg_data=esg_data,
        pdf_path=pdf_path,
        csv_path=csv_path,
    )


async def _run_audit_background(
    audit_id: str,
    company_name: str,
    esg_data: Optional[Dict[str, Any]],
    pdf_path: Optional[str],
    csv_path: Optional[str],
    fetch_news: bool,
    ml_weight: float,
    nlp_weight: float,
) -> None:
    """
    FastAPI BackgroundTask fallback when Celery is unavailable.
    Runs audit and persists result to the database.
    """
    from core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            await update_audit_record(
                db,
                audit_id,
                status=AuditStatus.RUNNING,
                started_at=datetime.utcnow(),
            )

            result = _execute_audit_pipeline(
                audit_id=audit_id,
                company_name=company_name,
                esg_data=esg_data,
                pdf_path=pdf_path,
                csv_path=csv_path,
                fetch_news=fetch_news,
                ml_weight=ml_weight,
                nlp_weight=nlp_weight,
            )

            fusion = result.get("fusion", {})
            ml_risk = result.get("ml_risk", {})
            validation = result.get("validation", {})
            report = result.get("report", {})

            await update_audit_record(
                db,
                audit_id,
                status=AuditStatus.COMPLETED,
                completed_at=datetime.utcnow(),
                composite_score=fusion.get("composite_esg_score"),
                risk_tier=fusion.get("risk_tier"),
                ml_risk_score=ml_risk.get("risk_score_ml"),
                validation_status=validation.get("validation_status"),
                result_json=result,
                json_report_path=report.get("json_report_path"),
                markdown_report_path=report.get("markdown_report_path"),
            )
            await db.commit()

        except Exception as exc:
            logger.error(
                "Background audit failed",
                extra={"audit_id": audit_id, "error": str(exc)},
                exc_info=True,
            )
            try:
                await update_audit_record(
                    db,
                    audit_id,
                    status=AuditStatus.FAILED,
                    completed_at=datetime.utcnow(),
                    error_message=str(exc)[:1000],
                )
                await db.commit()
            except Exception:
                pass


def _build_audit_result(
    audit_id: str,
    company_name: str,
    pipeline_result: Dict[str, Any],
    record: Optional[AuditRecord] = None,
) -> AuditResult:
    """Construct an AuditResult from raw pipeline output."""
    fusion = pipeline_result.get("fusion", {})
    ml_risk = pipeline_result.get("ml_risk", {})
    zero_shot = pipeline_result.get("zero_shot", {})
    validation = pipeline_result.get("validation", {})
    report = pipeline_result.get("report", {})

    # Dimensional scores
    dim_scores = fusion.get("dimensional_scores")

    # Emerging risks
    emerging = [
        EmergingRisk(**r)
        for r in fusion.get("emerging_risks", [])[:5]
        if isinstance(r, dict)
    ]

    # Risk signals
    signals = [
        RiskSignal(**s)
        for s in fusion.get("risk_signals", [])
        if isinstance(s, dict)
    ]

    # Regulatory flags
    reg_checks = validation.get("regulatory_checks", {})
    regulatory_flags = {}
    for framework_key, flags in reg_checks.items():
        if isinstance(flags, list):
            regulatory_flags[framework_key] = [
                ValidationFlag(**f) for f in flags if isinstance(f, dict)
            ]

    # Feature importances
    fi = ml_risk.get("feature_importances", {})

    # SHAP contributions
    shap_data = validation.get("shap_values", {})
    shap_contributions = None
    if isinstance(shap_data.get("contributions"), dict):
        from api.schemas import SHAPContribution
        shap_contributions = {
            k: SHAPContribution(**v)
            for k, v in shap_data["contributions"].items()
            if isinstance(v, dict)
        }

    from api.schemas import ConfidenceInterval
    ci_data = fusion.get("confidence_interval", {})
    ci = ConfidenceInterval(**ci_data) if ci_data else None

    nlp_agg = zero_shot.get("aggregate_scores", {})

    return AuditResult(
        audit_id=audit_id,
        company_name=company_name,
        status=AuditStatusEnum.COMPLETED,
        created_at=record.created_at if record else None,
        completed_at=record.completed_at if record else None,
        duration_seconds=record.duration_seconds if record else pipeline_result.get("duration_seconds"),
        composite_esg_score=fusion.get("composite_esg_score"),
        risk_tier=fusion.get("risk_tier"),
        risk_tier_description=fusion.get("risk_tier_description"),
        dimensional_scores=dim_scores,
        confidence_interval=ci,
        ml_risk_score=ml_risk.get("risk_score_ml"),
        ml_risk_tier=ml_risk.get("risk_tier_ml"),
        ml_prediction_confidence=ml_risk.get("prediction_confidence"),
        feature_importances=fi,
        top_risk_drivers=ml_risk.get("top_risk_drivers"),
        nlp_aggregate_scores={k: float(v) for k, v in nlp_agg.items()},
        emerging_risks=emerging,
        texts_analyzed=zero_shot.get("texts_analyzed"),
        validation_status=validation.get("validation_status"),
        regulatory_flags=regulatory_flags,
        total_flags=validation.get("total_flags"),
        shap_contributions=shap_contributions,
        lime_explanation=validation.get("lime_explanation"),
        risk_signals=signals,
        investment_recommendation=fusion.get("investment_recommendation"),
        executive_summary=report.get("executive_summary"),
        json_report_path=report.get("json_report_path"),
        markdown_report_path=report.get("markdown_report_path"),
        pdf_report_path=report.get("pdf_report_path"),
        error_message=pipeline_result.get("error"),
    )