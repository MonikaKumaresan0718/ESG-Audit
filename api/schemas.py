"""
Pydantic schemas for ESG Auditor API request and response validation.
All schemas follow OpenAPI 3.1 conventions.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# ── Enums ────────────────────────────────────────────────────────────────────

class RiskTierEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class AuditStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationStatusEnum(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


# ── Request Schemas ───────────────────────────────────────────────────────────

class ESGDataInput(BaseModel):
    """
    Optional structured ESG metrics to include in the audit.
    All fields are optional; missing fields use model defaults.
    """

    carbon_emissions: Optional[float] = Field(
        None,
        ge=0,
        le=10000,
        description="Annual carbon emissions in metric tons CO2e",
        example=245.7,
    )
    water_usage: Optional[float] = Field(
        None,
        ge=0,
        le=100000,
        description="Annual water usage in million liters",
        example=512.3,
    )
    board_diversity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Proportion of diverse board members (0–1)",
        example=0.42,
    )
    employee_turnover: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Annual employee turnover rate (0–1)",
        example=0.15,
    )
    controversy_score: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="ESG controversy score (0 = no controversy, 10 = severe)",
        example=3.2,
    )
    renewable_energy_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Proportion of energy from renewable sources (0–1)",
        example=0.35,
    )
    supply_chain_risk: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Supply chain ESG risk score (0–10)",
        example=4.8,
    )


class AuditRequest(BaseModel):
    """
    Request body for triggering an ESG audit.
    """

    company_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the company to audit",
        example="Tesla Inc.",
    )
    esg_data: Optional[ESGDataInput] = Field(
        None,
        description="Optional pre-loaded structured ESG metrics",
    )
    pdf_path: Optional[str] = Field(
        None,
        description="Path to a sustainability report PDF (server-side path)",
        example="/data/reports/tesla_2024_sustainability.pdf",
    )
    csv_path: Optional[str] = Field(
        None,
        description="Path to an ESG metrics CSV file (server-side path)",
        example="/data/sample_esg_data.csv",
    )
    fetch_news: bool = Field(
        False,
        description="Whether to fetch recent news articles for NLP analysis",
    )
    async_execution: bool = Field(
        True,
        description="If True, runs audit asynchronously via Celery; False runs inline",
    )
    ml_weight: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Weight assigned to ML model in hybrid fusion (0–1)",
    )
    nlp_weight: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Weight assigned to NLP model in hybrid fusion (0–1)",
    )

    @validator("nlp_weight")
    def weights_must_sum_to_one(cls, v: float, values: dict) -> float:
        ml_w = values.get("ml_weight", 0.6)
        if abs(ml_w + v - 1.0) > 0.01:
            raise ValueError(
                f"ml_weight ({ml_w}) + nlp_weight ({v}) must sum to 1.0"
            )
        return v

    @validator("company_name")
    def company_name_no_special_chars(cls, v: str) -> str:
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "Tesla Inc.",
                "esg_data": {
                    "carbon_emissions": 245.7,
                    "water_usage": 512.3,
                    "board_diversity": 0.42,
                    "employee_turnover": 0.15,
                    "controversy_score": 3.2,
                    "renewable_energy_pct": 0.35,
                    "supply_chain_risk": 4.8,
                },
                "fetch_news": False,
                "async_execution": True,
                "ml_weight": 0.6,
                "nlp_weight": 0.4,
            }
        }


# ── Response Schemas ──────────────────────────────────────────────────────────

class DimensionalScore(BaseModel):
    """E, S, or G dimensional score breakdown."""
    score: float = Field(..., ge=0, le=100)
    ml_component: float
    nlp_component: float


class DimensionalScores(BaseModel):
    """ESG dimensional score breakdown."""
    environmental: DimensionalScore
    social: DimensionalScore
    governance: DimensionalScore


class ConfidenceInterval(BaseModel):
    """Confidence interval around a score estimate."""
    lower: float
    upper: float
    uncertainty: float


class EmergingRisk(BaseModel):
    """A detected emerging ESG risk from NLP analysis."""
    risk: str
    confidence: float
    source_text_index: Optional[int] = None
    text_excerpt: Optional[str] = None


class RiskSignal(BaseModel):
    """A specific risk signal identified during the audit."""
    type: str
    message: str
    severity: str


class FeatureImportance(BaseModel):
    """ML model feature importance entry."""
    feature: str
    importance: float


class SHAPContribution(BaseModel):
    """SHAP value contribution for a feature."""
    value: float
    shap_value: float
    importance: float


class ValidationFlag(BaseModel):
    """Regulatory validation flag."""
    framework: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    severity: str
    message: str


class AuditSummary(BaseModel):
    """
    Lightweight audit summary returned in list views and async responses.
    """
    audit_id: str
    company_name: str
    status: AuditStatusEnum
    composite_score: Optional[float] = None
    risk_tier: Optional[RiskTierEnum] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class AuditResult(BaseModel):
    """
    Full ESG audit result returned by GET /v1/audit/{audit_id}.
    """

    audit_id: str = Field(..., description="Unique audit identifier")
    company_name: str
    status: AuditStatusEnum
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Core scores
    composite_esg_score: Optional[float] = Field(None, ge=0, le=100)
    risk_tier: Optional[RiskTierEnum] = None
    risk_tier_description: Optional[str] = None
    dimensional_scores: Optional[Dict[str, Any]] = None
    confidence_interval: Optional[ConfidenceInterval] = None

    # ML analysis
    ml_risk_score: Optional[float] = None
    ml_risk_tier: Optional[str] = None
    ml_prediction_confidence: Optional[float] = None
    feature_importances: Optional[Dict[str, float]] = None
    top_risk_drivers: Optional[List[str]] = None

    # NLP analysis
    nlp_aggregate_scores: Optional[Dict[str, float]] = None
    emerging_risks: Optional[List[EmergingRisk]] = None
    texts_analyzed: Optional[int] = None

    # Validation
    validation_status: Optional[ValidationStatusEnum] = None
    regulatory_flags: Optional[Dict[str, List[ValidationFlag]]] = None
    total_flags: Optional[int] = None

    # SHAP / LIME
    shap_contributions: Optional[Dict[str, SHAPContribution]] = None
    lime_explanation: Optional[Dict[str, Any]] = None

    # Report paths
    risk_signals: Optional[List[RiskSignal]] = None
    investment_recommendation: Optional[str] = None
    executive_summary: Optional[str] = None

    # Report download paths
    json_report_path: Optional[str] = None
    markdown_report_path: Optional[str] = None
    pdf_report_path: Optional[str] = None

    # Error
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class AuditCreateResponse(BaseModel):
    """
    Response returned immediately after POST /v1/audit.
    For async audits, contains the audit_id for polling.
    """
    audit_id: str
    status: AuditStatusEnum
    message: str
    poll_url: Optional[str] = None
    result: Optional[AuditResult] = None


class HealthResponse(BaseModel):
    """Response for GET /healthz."""
    status: str = "ok"
    version: str
    environment: str
    database: str = "ok"
    redis: str = "unknown"
    model_loaded: bool = False
    uptime_seconds: Optional[float] = None


class ReportDownloadResponse(BaseModel):
    """Response for report download endpoints."""
    audit_id: str
    company_name: str
    format: str
    file_path: str
    file_size_bytes: Optional[int] = None
    generated_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    audit_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Audit not found",
                "detail": "No audit record found with id=abc-123",
                "audit_id": "abc-123",
            }
        }


class PaginatedAuditList(BaseModel):
    """Paginated list of audit summaries."""
    items: List[AuditSummary]
    total: int
    limit: int
    offset: int
    has_more: bool