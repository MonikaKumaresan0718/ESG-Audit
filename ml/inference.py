"""
ESG Risk Model Inference Engine.
Supports both batch and online inference with model auto-training fallback.
"""

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class ESGInference:
    """
    Production inference engine for ESG risk scoring.
    Auto-trains model on first use if no pre-trained artifact exists.
    """

    MODEL_VERSION = "xgb_v1"

    def __init__(
        self,
        model_path: Optional[str] = None,
        pipeline_path: Optional[str] = None,
    ):
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline_path = pipeline_path or settings.PIPELINE_PATH
        self._model = None
        self._pipeline = None

    def _load_or_train(self):
        """Load model from disk or train a new one if not available."""
        model_exists = self.model_path and os.path.exists(self.model_path)
        pipeline_exists = self.pipeline_path and os.path.exists(self.pipeline_path)

        if model_exists and pipeline_exists:
            try:
                self._model = joblib.load(self.model_path)
                self._pipeline = joblib.load(self.pipeline_path)
                logger.info(f"Loaded model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Retraining.")

        # Auto-train
        logger.info("No pre-trained model found. Training new model...")
        try:
            from ml.train import train_pipeline

            result = train_pipeline(save=True)
            self._model = result["model"]
            self._pipeline = result["feature_engineer"]
            logger.info("Model trained and loaded successfully")
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")
            self._model = None
            self._pipeline = None

    @property
    def model(self):
        if self._model is None:
            self._load_or_train()
        return self._model

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_or_train()
        return self._pipeline

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Run prediction on engineered features.

        Args:
            features: Feature DataFrame from ESGFeatureEngineer.

        Returns:
            Dict with predicted_class, probabilities, confidence, feature_importances.
        """
        model = self.model

        if model is None:
            return self._heuristic_prediction(features)

        try:
            # Align features with model expectations
            features_aligned = self._align_features(features, model)

            # Predict
            predicted_class = int(model.predict(features_aligned)[0])

            probabilities = []
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_aligned)[0]
                probabilities = proba.tolist()
            else:
                proba = np.zeros(3)
                proba[predicted_class] = 1.0
                probabilities = proba.tolist()

            # Confidence = max probability
            confidence = float(max(probabilities))

            # Feature importances
            importances = {}
            if hasattr(model, "feature_importances_"):
                feat_names = features_aligned.columns.tolist()
                raw_importances = model.feature_importances_
                importances = dict(zip(feat_names, raw_importances.tolist()))
                importances = {
                    k: round(float(v), 4)
                    for k, v in sorted(
                        importances.items(), key=lambda x: x[1], reverse=True
                    )
                }

            return {
                "predicted_class": predicted_class,
                "probabilities": probabilities,
                "confidence": round(confidence, 4),
                "feature_importances": importances,
                "model_version": self.MODEL_VERSION,
            }

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return self._heuristic_prediction(features)

    def predict_batch(self, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Run batch prediction on multiple records.

        Args:
            features_df: DataFrame with one row per company.

        Returns:
            List of prediction dicts.
        """
        results = []
        for _, row in features_df.iterrows():
            single_df = pd.DataFrame([row])
            results.append(self.predict(single_df))
        return results

    def _align_features(self, features: pd.DataFrame, model) -> pd.DataFrame:
        """Align feature DataFrame with model's expected features."""
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = set(expected) - set(features.columns)
            for col in missing:
                features[col] = 0.0
            return features[expected]

        # For XGBoost with booster
        if hasattr(model, "get_booster"):
            try:
                expected = model.get_booster().feature_names
                if expected:
                    missing = set(expected) - set(features.columns)
                    for col in missing:
                        features[col] = 0.0
                    return features[expected]
            except Exception:
                pass

        return features

    def _heuristic_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Fallback heuristic prediction when model unavailable."""
        if features.empty:
            return {
                "predicted_class": 1,
                "probabilities": [0.33, 0.34, 0.33],
                "confidence": 0.34,
                "feature_importances": {},
                "model_version": "heuristic",
            }

        row = features.iloc[0]
        risk_proxy = float(row.get("overall_risk_proxy", 0.4))

        if risk_proxy < 0.33:
            predicted_class = 0
            probs = [0.6, 0.3, 0.1]
        elif risk_proxy < 0.66:
            predicted_class = 1
            probs = [0.2, 0.6, 0.2]
        else:
            predicted_class = 2
            probs = [0.1, 0.3, 0.6]

        return {
            "predicted_class": predicted_class,
            "probabilities": probs,
            "confidence": float(max(probs)),
            "feature_importances": {
                "overall_risk_proxy": 0.25,
                "carbon_emissions": 0.20,
                "controversy_score": 0.18,
                "employee_turnover": 0.15,
                "board_diversity": 0.12,
            },
            "model_version": "heuristic_fallback",
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        model = self.model
        if model is None:
            return {"status": "no_model", "version": self.MODEL_VERSION}

        info = {
            "version": self.MODEL_VERSION,
            "model_class": type(model).__name__,
            "model_path": self.model_path,
            "status": "loaded",
        }

        if hasattr(model, "n_estimators"):
            info["n_estimators"] = model.n_estimators
        if hasattr(model, "feature_names_in_"):
            info["n_features"] = len(model.feature_names_in_)

        return info