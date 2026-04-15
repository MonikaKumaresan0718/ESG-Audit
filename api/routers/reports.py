"""
Reports router – GET /v1/report/{audit_id}/{format}
Serves generated ESG audit reports for download.
"""

import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import ReportDownloadResponse
from core.database import get_audit_record, get_db, AuditStatus
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/report", tags=["Reports"])

SUPPORTED_FORMATS = {"json", "markdown", "pdf", "md"}


@router.get(
    "/{audit_id}/json",
    summary="Download JSON Report",
    description="Download the full ESG audit report in JSON format.",
)
async def download_json_report(
    audit_id: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """GET /v1/report/{audit_id}/json"""
    return await _serve_report(audit_id, "json", db)


@router.get(
    "/{audit_id}/markdown",
    summary="Download Markdown Report",
    description="Download the ESG audit report in Markdown format.",
)
async def download_markdown_report(
    audit_id: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """GET /v1/report/{audit_id}/markdown"""
    return await _serve_report(audit_id, "markdown", db)


@router.get(
    "/{audit_id}/pdf",
    summary="Download PDF Report",
    description="Download the ESG audit report as a PDF file.",
)
async def download_pdf_report(
    audit_id: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """GET /v1/report/{audit_id}/pdf"""
    return await _serve_report(audit_id, "pdf", db)


@router.get(
    "/{audit_id}/info",
    response_model=ReportDownloadResponse,
    summary="Report Metadata",
    description="Get metadata about available reports for an audit.",
)
async def get_report_info(
    audit_id: str,
    db: AsyncSession = Depends(get_db),
) -> ReportDownloadResponse:
    """GET /v1/report/{audit_id}/info"""
    record = await get_audit_record(db, audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found")

    if record.status != AuditStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Audit is not yet completed (status: {record.status.value})",
        )

    # Determine best available format
    path = record.json_report_path or record.markdown_report_path
    fmt = "json" if record.json_report_path else "markdown"
    file_size = None

    if path and os.path.exists(path):
        file_size = os.path.getsize(path)

    return ReportDownloadResponse(
        audit_id=audit_id,
        company_name=record.company_name,
        format=fmt,
        file_path=path or "",
        file_size_bytes=file_size,
        generated_at=record.completed_at,
    )


# ── Internal helper ───────────────────────────────────────────────────────────

async def _serve_report(
    audit_id: str,
    fmt: str,
    db: AsyncSession,
) -> FileResponse:
    """Resolve and serve a report file."""
    record = await get_audit_record(db, audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Audit '{audit_id}' not found")

    if record.status != AuditStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Report not available yet. Audit status: {record.status.value}. "
                "Please poll GET /v1/audit/{audit_id} until status is 'completed'."
            ),
        )

    # Resolve file path by format
    path_map = {
        "json": record.json_report_path,
        "markdown": record.markdown_report_path,
        "md": record.markdown_report_path,
        "pdf": record.pdf_report_path,
    }

    media_map = {
        "json": "application/json",
        "markdown": "text/markdown",
        "md": "text/markdown",
        "pdf": "application/pdf",
    }

    suffix_map = {
        "json": ".json",
        "markdown": ".md",
        "md": ".md",
        "pdf": ".pdf",
    }

    file_path = path_map.get(fmt)

    if not file_path:
        # Attempt to regenerate on-the-fly for JSON from stored result_json
        if fmt == "json" and record.result_json:
            import json
            from core.config import settings

            os.makedirs(settings.REPORTS_DIR, exist_ok=True)
            tmp_path = os.path.join(settings.REPORTS_DIR, f"esg_audit_{audit_id}.json")
            with open(tmp_path, "w") as f:
                json.dump(record.result_json, f, indent=2, default=str)
            file_path = tmp_path
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Report in '{fmt}' format not available for audit '{audit_id}'",
            )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found on disk. It may have been deleted.",
        )

    filename = f"ESG_Audit_{record.company_name.replace(' ', '_')}_{audit_id[:8]}{suffix_map.get(fmt, '.txt')}"

    logger.info(
        f"Serving {fmt} report",
        extra={"audit_id": audit_id, "format": fmt, "path": file_path},
    )

    return FileResponse(
        path=file_path,
        media_type=media_map.get(fmt, "application/octet-stream"),
        filename=filename,
    )