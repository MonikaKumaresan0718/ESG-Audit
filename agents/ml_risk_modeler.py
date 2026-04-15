"""
MLRiskModelerAgent – Implements ESG feature engineering, trains/loads an XGBoost
model, predicts ESG risk scores, and provides feature importance using joblib.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from crewai import Agent

from core.config import settings
from core.logging import get_logger
from ml.feature_engineering import ESGFeatureEngineer
from ml.inference import ESGInference

logger = get_logger(__name__)


class MLRiskModelerAgent:
    """
    ML-based ESG risk modeling agent.
    Trains and deploys XGBoost/Scikit-learn models for structured ESG risk scoring.
    """

    def __init__(self):
        self.feature_engineer = ESGFeatureEngineer()
        self.inference_engine = ESGInference()
        self.agent = self._build_crewai_agent()

    def _build_crewai_agent(self) -> Agent:
        return Agent(
            role="ML ESG Risk Modeler",
            goal=(
                "Engineer ESG features from structured data, load or train an XGBoost "
                "risk scoring model, generate risk predictions with confidence scores, "
                "and provide feature importance explanations for audit transparency."
            ),
            backstory=(
                "You are a quantitative analyst and ML engineer who has built risk "
                "scoring models for leading ESG rating agencies including MSCI and "
                "Sustainalytics. You specialize in gradient boosting methods and "
                "feature engineering for tabular ESG data. Your models are trusted "
                "by institutional investors for portfolio risk assessment."
            ),
            verbose=settings.VERBOSE_AGENTS,
            allow_delegation=False,
        )

    def predict(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict ESG risk score from structured data.

        Args:
            structured_data: Dict of ESG metrics.

        Returns:
            Dict with risk score, tier, probabilities, and feature importances.
        """
        logger.info("Running ML risk prediction")

        if not structured_data:
            logger.warning("No structured data provided; using synthetic defaults")
            structured_data = self._default_esg_data()

        try:
            # Feature engineering
            features_df = self.feature_engineer.transform(structured_data)
            feature_names = features_df.columns.tolist()

            # Run inference
            prediction = self.inference_engine.predict(features_df)

            # Compute risk score (0–100)
            risk_score = self._compute_risk_score(
                structured_data, prediction.get("probabilities", [0.33, 0.33, 0.34])
            )

            # Assign risk tier
            risk_tier = self._assign_risk_tier(risk_score)

            # Feature importances
            importances = prediction.get("feature_importances", {})
            if not importances and feature_names:
                importances = self._compute_heuristic_importances(
                    structured_data, feature_names
                )

            return {
                "risk_score_ml": round(float(risk_score), 2),
                "risk_tier_ml": risk_tier,
                "predicted_class": prediction.get("predicted_class", 1),
                "class_probabilities": {
                    "low": round(float(prediction.get("probabilities", [0.4, 0.35, 0.25])[0]), 4),
                    "medium": round(float(prediction.get("probabilities", [0.4, 0.35, 0.25])[1]), 4),
                    "high": round(float(prediction.get("probabilities", [0.4, 0.35, 0.25])[2]), 4),
                },
                "feature_importances": importances,
                "features_used": feature_names,
                "model_version": prediction.get("model_version", "xgb_v1"),
                "prediction_confidence": prediction.get("confidence", 0.75),
                "input_data_summary": self._summarize_inputs(structured_data),
            }

        except Exception as e:
            logger.error(f"ML prediction failed: {e}", exc_info=True)
            return self._fallback_prediction(structured_data)

    def _compute_risk_score(
        self, data: Dict[str, Any], probabilities: List[float]
    ) -> float:
        """
        Compute a composite ESG risk score (0–100) from probabilities
        and raw ESG metrics.
        """
        # Weighted probability score (higher probability of high risk → higher score)
        prob_score = 0.0
        if len(probabilities) >= 3:
            prob_score = (
                probabilities[0] * 20 +    # low risk → low score
                probabilities[1] * 50 +    # medium
                probabilities[2] * 80      # high risk → high score
            )
        elif len(probabilities) == 2:
            prob_score = probabilities[1] * 80

        # Rule-based score from metrics
        rule_score = 0.0
        carbon = float(data.get("carbon_emissions", 200))
        water = float(data.get("water_usage", 500))
        diversity = float(data.get("board_diversity", 0.4))
        turnover = float(data.get("employee_turnover", 0.15))
        controversy = float(data.get("controversy_score", 5))

        # Normalize and invert where necessary
        rule_score += min(carbon / 500 * 25, 25)    # max 25 pts for carbon
        rule_score += min(water / 1000 * 15, 15)    # max 15 pts for water
        rule_score += max(0, (0.5 - diversity) * 40) # lower diversity → higher score
        rule_score += min(turnover / 0.3 * 20, 20)   # higher turnover → higher score
        rule_score += min(controversy / 10 * 25, 25) # controversy score

        # Blend probability + rule
        final_score = 0.6 * prob_score + 0.4 * rule_score
        return float(np.clip(final_score, 0, 100))

    def _assign_risk_tier(self, score: float) -> str:
        """Assign risk tier based on composite score."""
        if score < 25:
            return "LOW"
        elif score < 50:
            return "MEDIUM"
        elif score < 75:
            return "HIGH"
        else:
            return "CRITICAL"

    def _compute_heuristic_importances(
        self, data: Dict[str, Any], feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute heuristic feature importances as fallback."""
        importance_weights = {
            "carbon_emissions": 0.22,
            "controversy_score": 0.20,
            "water_usage": 0.15,
            "employee_turnover": 0.15,
            "board_diversity": 0.13,
            "renewable_energy_pct": 0.08,
            "supply_chain_risk": 0.07,
        }
        result = {}
        for fname in feature_names:
            base = fname.split("__")[-1] if "__" in fname else fname
            result[fname] = round(importance_weights.get(base, 0.05), 4)

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total, 4) for k, v in result.items()}

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def _summarize_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize input ESG data for audit trail."""
        numeric_keys = [
            "carbon_emissions", "water_usage", "board_diversity",
            "employee_turnover", "controversy_score", "renewable_energy_pct",
            "supply_chain_risk",
        ]
        return {k: round(float(data[k]), 4) for k in numeric_keys if k in data}

    def _default_esg_data(self) -> Dict[str, Any]:
        """Generate default ESG data for testing."""
        return {
            "carbon_emissions": 250.0,
            "water_usage": 500.0,
            "board_diversity": 0.35,
            "employee_turnover": 0.18,
            "controversy_score": 4.5,
            "renewable_energy_pct": 0.3,
            "supply_chain_risk": 5.0,
            "esg_risk_label": 1,
        }

    def _fallback_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return heuristic prediction when ML model fails."""
        risk_score = self._compute_risk_score(data, [0.33, 0.33, 0.34])
        risk_tier = self._assign_risk_tier(risk_score)

        return {
            "risk_score_ml": round(float(risk_score), 2),
            "risk_tier_ml": risk_tier,
            "predicted_class": 1,
            "class_probabilities": {"low": 0.33, "medium": 0.34, "high": 0.33},
            "feature_importances": self._compute_heuristic_importances(
                data, list(data.keys())
            ),
            "features_used": list(data.keys()),
            "model_version": "heuristic_fallback",
            "prediction_confidence": 0.5,
            "input_data_summary": self._summarize_inputs(data),
        }