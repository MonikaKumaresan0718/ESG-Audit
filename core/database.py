"""
SQLAlchemy async database engine and ORM models for ESG Auditor.
Uses aiosqlite for SQLite in development; swap DATABASE_URL for PostgreSQL in prod.
"""

import enum
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# ── Engine ───────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Base Model ───────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ── Enums ────────────────────────────────────────────────────────────────────

class AuditStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskTier(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


# ── ORM Models ───────────────────────────────────────────────────────────────

class AuditRecord(Base):
    """
    Stores ESG audit requests and their results.
    One record per audit invocation.
    """

    __tablename__ = "audit_records"

    id = Column(String(36), primary_key=True, index=True)
    company_name = Column(String(255), nullable=False, index=True)
    status = Column(
        Enum(AuditStatus),
        default=AuditStatus.PENDING,
        nullable=False,
        index=True,
    )
    celery_task_id = Column(String(255), nullable=True)

    # Results
    composite_score = Column(Float, nullable=True)
    risk_tier = Column(String(20), nullable=True)
    ml_risk_score = Column(Float, nullable=True)
    validation_status = Column(String(20), nullable=True)

    # Full pipeline output stored as JSON
    result_json = Column(JSON, nullable=True)

    # Report paths
    json_report_path = Column(String(512), nullable=True)
    markdown_report_path = Column(String(512), nullable=True)
    pdf_report_path = Column(String(512), nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<AuditRecord id={self.id!r} company={self.company_name!r} "
            f"status={self.status!r}>"
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Compute audit duration if both timestamps are available."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class ModelMetrics(Base):
    """
    Tracks ML model training runs and evaluation metrics.
    Used for model versioning and drift monitoring.
    """

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    cv_f1_mean = Column(Float, nullable=True)
    cv_f1_std = Column(Float, nullable=True)
    n_training_samples = Column(Integer, nullable=True)
    n_features = Column(Integer, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    feature_importances = Column(JSON, nullable=True)
    trained_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ModelMetrics version={self.model_version!r} "
            f"f1={self.f1_score:.4f} trained_at={self.trained_at!r}>"
        )


# ── Session Helpers ───────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session.
    Ensures the session is properly closed after each request.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables() -> None:
    """Create all database tables. Called at application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")


async def drop_tables() -> None:
    """Drop all database tables. Used in testing."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped")


# ── CRUD Helpers ──────────────────────────────────────────────────────────────

async def create_audit_record(
    db: AsyncSession,
    audit_id: str,
    company_name: str,
    celery_task_id: Optional[str] = None,
) -> AuditRecord:
    """Insert a new pending audit record."""
    record = AuditRecord(
        id=audit_id,
        company_name=company_name,
        status=AuditStatus.PENDING,
        celery_task_id=celery_task_id,
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)
    return record


async def get_audit_record(
    db: AsyncSession, audit_id: str
) -> Optional[AuditRecord]:
    """Fetch an audit record by ID."""
    from sqlalchemy import select

    result = await db.execute(
        select(AuditRecord).where(AuditRecord.id == audit_id)
    )
    return result.scalar_one_or_none()


async def update_audit_record(
    db: AsyncSession,
    audit_id: str,
    **kwargs,
) -> Optional[AuditRecord]:
    """Update fields on an existing audit record."""
    from sqlalchemy import update

    kwargs["updated_at"] = datetime.utcnow()
    await db.execute(
        update(AuditRecord)
        .where(AuditRecord.id == audit_id)
        .values(**kwargs)
    )
    await db.flush()
    return await get_audit_record(db, audit_id)


async def list_audit_records(
    db: AsyncSession,
    limit: int = 50,
    offset: int = 0,
    company_filter: Optional[str] = None,
) -> list:
    """List audit records with optional company filter."""
    from sqlalchemy import select

    query = select(AuditRecord).order_by(AuditRecord.created_at.desc())

    if company_filter:
        query = query.where(
            AuditRecord.company_name.ilike(f"%{company_filter}%")
        )

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    return list(result.scalars().all())