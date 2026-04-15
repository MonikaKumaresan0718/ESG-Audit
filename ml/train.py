"""
ESG Risk Model Training Entry Point.
Trains XGBoost/Scikit-learn classifier and saves artifacts using joblib.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from core.config import settings
from core.logging import get_logger
from ml.feature_engineering import ESGFeatureEngineer, load_and_prepare_dataset

logger = get_logger(__name__)


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: Optional[Dict[str, Any]] = None,
):
    """Train XGBoost classifier for ESG risk scoring."""
    try:
        from xgboost import XGBClassifier

        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        if hyperparams:
            params.update(hyperparams)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost model trained successfully")
        return model

    except ImportError:
        logger.warning("XGBoost not available; falling back to GradientBoosting")
        return train_sklearn_model(X_train, y_train, hyperparams)


def train_sklearn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: Optional[Dict[str, Any]] = None,
):
    """Train Scikit-learn GradientBoosting classifier as fallback."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier

    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    }
    if hyperparams:
        params.update(hyperparams)

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("GradientBoosting model trained successfully")
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Evaluate trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)
        if hasattr(model, "predict_proba")
        else None
    )

    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics: Dict[str, Any] = {
        "f1_weighted": round(float(f1), 4),
        "classification_report": report,
        "n_test_samples": len(y_test),
    }

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                )
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            pass

    # Cross-validation
    if X_train is not None and y_train is not None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
        metrics["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_f1_std"] = round(float(cv_scores.std()), 4)

    logger.info(f"Model evaluation: F1={f1:.4f}")
    return metrics


def get_feature_importances(model, feature_names: list) -> Dict[str, float]:
    """Extract feature importances from trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_dict = dict(zip(feature_names, importances))
        return dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
    return {}


def save_artifacts(
    model,
    feature_engineer: ESGFeatureEngineer,
    metrics: Dict[str, Any],
    model_path: Optional[str] = None,
    pipeline_path: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Save model and feature pipeline artifacts.

    Returns:
        Paths to saved model and pipeline files.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = model_path or os.path.join(model_dir, "esg_xgb_v1.pkl")
    pipeline_path = pipeline_path or os.path.join(model_dir, "feature_pipeline.pkl")

    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature engineer
    joblib.dump(feature_engineer, pipeline_path)
    logger.info(f"Feature pipeline saved to {pipeline_path}")

    # Save metrics as JSON
    import json

    metrics_path = os.path.join(model_dir, "training_metrics.json")
    metrics["trained_at"] = datetime.utcnow().isoformat()
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return model_path, pipeline_path


def train_pipeline(
    csv_path: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    model_type: str = "xgboost",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Full training pipeline.

    Args:
        csv_path: Path to ESG CSV data. Defaults to sample data.
        hyperparams: Model hyperparameters dict.
        model_type: 'xgboost' or 'sklearn'.
        save: Whether to save artifacts.

    Returns:
        Training result dict with model, metrics, and paths.
    """
    # Load data
    csv_path = csv_path or os.path.join(settings.DATA_DIR, "sample_esg_data.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found at {csv_path}; generating synthetic data")
        csv_path = _generate_synthetic_data(csv_path)

    X_train, X_test, y_train, y_test = load_and_prepare_dataset(csv_path)
    feature_names = X_train.columns.tolist()

    logger.info(
        f"Training data: {X_train.shape[0]} samples, {len(feature_names)} features"
    )

    # Train model
    if model_type == "xgboost":
        model = train_xgboost_model(X_train, y_train, hyperparams)
    else:
        model = train_sklearn_model(X_train, y_train, hyperparams)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)
    feature_importances = get_feature_importances(model, feature_names)
    metrics["feature_importances"] = feature_importances

    # Feature engineer
    engineer = ESGFeatureEngineer()

    # Save
    model_path, pipeline_path = None, None
    if save:
        model_path, pipeline_path = save_artifacts(model, engineer, metrics)

    return {
        "model": model,
        "feature_engineer": engineer,
        "metrics": metrics,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "n_classes": len(y_train.unique()),
    }


def _generate_synthetic_data(save_path: str) -> str:
    """Generate and save synthetic ESG training data."""
    from data import generate_sample_esg_data

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = generate_sample_esg_data(n_samples=200)
    df.to_csv(save_path, index=False)
    logger.info(f"Synthetic data saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESG Risk Model")
    parser.add_argument("--csv", type=str, default=None, help="Path to ESG CSV data")
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "sklearn"],
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save artifacts")
    args = parser.parse_args()

    result = train_pipeline(
        csv_path=args.csv,
        model_type=args.model_type,
        save=not args.no_save,
    )

    print(f"\n✅ Training Complete!")
    print(f"F1 Score: {result['metrics']['f1_weighted']:.4f}")
    if "roc_auc" in result["metrics"]:
        print(f"ROC-AUC: {result['metrics']['roc_auc']:.4f}")
    if "cv_f1_mean" in result["metrics"]:
        print(f"CV F1: {result['metrics']['cv_f1_mean']:.4f} ± {result['metrics']['cv_f1_std']:.4f}")
    print(f"\nTop Features:")
    for feat, imp in list(result["feature_importances"].items())[:5]:
        print(f"  {feat}: {imp:.4f}")