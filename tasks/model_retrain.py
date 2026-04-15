"""
Celery task: retrain_model_task
Scheduled weekly retraining of the ESG risk ML model.
Supports forced retraining and performance-gated updates.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Task

from core.celery_app import celery_app
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Minimum F1 improvement required to replace existing model
MIN_F1_IMPROVEMENT_THRESHOLD = 0.02


class RetrainTask(Task):
    """Custom base class for model retraining tasks."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            f"Model retraining task failed",
            extra={"celery_task_id": task_id, "error": str(exc)},
        )


@celery_app.task(
    bind=True,
    base=RetrainTask,
    name="tasks.model_retrain.retrain_model_task",
    max_retries=1,
    default_retry_delay=3600,
    soft_time_limit=1800,
    time_limit=2400,
)
def retrain_model_task(
    self: RetrainTask,
    force: bool = False,
    csv_path: Optional[str] = None,
    model_type: str = "xgboost",
    min_f1_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Celery task: Retrain the ESG risk ML model.

    Args:
        force: If True, replace model even if new F1 is lower.
        csv_path: Path to training data CSV. Defaults to sample data.
        model_type: 'xgboost' or 'sklearn'.
        min_f1_threshold: Minimum F1 score to deploy new model.

    Returns:
        Dict with training metrics and deployment decision.
    """
    logger.info(
        "Model retraining task started",
        extra={
            "celery_task_id": self.request.id,
            "force": force,
            "model_type": model_type,
        },
    )

    run_id = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    try:
        # ── Load existing model metrics if available ──────────────────────────
        existing_metrics = _load_existing_metrics()
        existing_f1 = existing_metrics.get("f1_weighted", 0.0)

        # ── Train new model ───────────────────────────────────────────────────
        from ml.train import train_pipeline

        logger.info(f"Training new {model_type} model [run_id={run_id}]")
        train_result = train_pipeline(
            csv_path=csv_path,
            model_type=model_type,
            save=False,  # Save conditionally below
        )

        new_metrics = train_result["metrics"]
        new_f1 = new_metrics.get("f1_weighted", 0.0)

        logger.info(
            f"New model trained: F1={new_f1:.4f}, existing F1={existing_f1:.4f}"
        )

        # ── Deployment decision ───────────────────────────────────────────────
        should_deploy = force or (
            new_f1 >= min_f1_threshold
            and (new_f1 - existing_f1) >= -MIN_F1_IMPROVEMENT_THRESHOLD
        )

        if should_deploy:
            # Backup existing model
            _backup_model()

            # Save new artifacts
            from ml.train import save_artifacts

            model_path, pipeline_path = save_artifacts(
                model=train_result["model"],
                feature_engineer=train_result["feature_engineer"],
                metrics=new_metrics,
            )

            # Log to experiments directory
            _log_experiment(
                run_id=run_id,
                metrics=new_metrics,
                model_type=model_type,
                deployed=True,
            )

            logger.info(
                "New model deployed",
                extra={
                    "run_id": run_id,
                    "new_f1": new_f1,
                    "model_path": model_path,
                },
            )

            return {
                "run_id": run_id,
                "status": "deployed",
                "new_f1": new_f1,
                "existing_f1": existing_f1,
                "improvement": round(new_f1 - existing_f1, 4),
                "model_path": model_path,
                "metrics": new_metrics,
            }

        else:
            reason = (
                f"New F1 ({new_f1:.4f}) below threshold ({min_f1_threshold}) "
                f"or insufficient improvement over existing ({existing_f1:.4f})"
            )
            logger.warning(f"New model NOT deployed: {reason}")

            _log_experiment(
                run_id=run_id,
                metrics=new_metrics,
                model_type=model_type,
                deployed=False,
                notes=reason,
            )

            return {
                "run_id": run_id,
                "status": "not_deployed",
                "reason": reason,
                "new_f1": new_f1,
                "existing_f1": existing_f1,
                "metrics": new_metrics,
            }

    except Exception as exc:
        logger.error(
            f"Model retraining failed: {exc}",
            extra={"run_id": run_id},
            exc_info=True,
        )
        raise


def _load_existing_metrics() -> Dict[str, Any]:
    """Load metrics from the most recently trained model."""
    metrics_path = os.path.join(settings.MODEL_DIR, "training_metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    try:
        import json

        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _backup_model() -> None:
    """Create a timestamped backup of the current model artifacts."""
    import shutil

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(settings.MODEL_DIR, f"backup_{ts}")

    if os.path.exists(settings.MODEL_PATH):
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(settings.MODEL_PATH, os.path.join(backup_dir, "esg_xgb_v1.pkl"))
        logger.info(f"Model backed up to {backup_dir}")

    if os.path.exists(settings.PIPELINE_PATH):
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(settings.PIPELINE_PATH, os.path.join(backup_dir, "feature_pipeline.pkl"))


def _log_experiment(
    run_id: str,
    metrics: Dict[str, Any],
    model_type: str,
    deployed: bool,
    notes: str = "",
) -> None:
    """Log experiment metadata to the experiments directory."""
    import json

    exp_dir = os.path.join(os.path.dirname(settings.MODEL_DIR), "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    exp_file = os.path.join(exp_dir, f"{run_id}.json")
    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_type": model_type,
        "deployed": deployed,
        "notes": notes,
        "metrics": metrics,
    }

    with open(exp_file, "w") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info(f"Experiment logged to {exp_file}")