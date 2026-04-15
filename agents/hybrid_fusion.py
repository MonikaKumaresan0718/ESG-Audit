"""
HybridFusionAgent – Combines ML predictions with zero-shot NLP insights using
a configurable weighted ensemble to produce a composite ESG score (0–100)
and assign a risk tier: LOW, MEDIUM, HIGH, or CRITICAL.
"""

from typing import Any, Dict, Optional

from crewai import Agent

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

RISK_TIER_THRESHOLDS = {
    "LOW": (0, 25),
    "MEDIUM": (25, 50),
    "HIGH": (50, 75),
    "CRITICAL": (75, 100),
}


class HybridFusionAgent:
    """
    Fuses ML risk scores and zero-shot NLP risk scores into a composite ESG score.
    Supports configurable weighting between ML and NLP contributions.
    """

    def __init__(
        self,
        ml_weight: float = 0.6,
        nlp_weight: float = 0.4,
    ):
        """
        Args:
            ml_weight: Weight assigned to ML model score (0–1).
            nlp_weight: Weight assigned to zero-shot NLP score (0–1).
        """
        assert abs(ml_weight + nlp_weight - 1.0) < 1e-6, (
            "Weights must sum to 1.0"
        )
        self.ml_weight = ml_weight
        self.nlp_weight = nlp_weight
        self.agent = self._build_crewai_agent()

    def _build_crewai_agent(self) -> Agent:
        return Agent(
            role="ESG Hybrid Fusion Analyst",
            goal=(
                "Combine structured ML risk scores with unstructured zero-shot NLP "
                "insights into a single composite ESG score (0–100) using a "
                "configurable weighted ensemble. Assign a risk tier and provide "
                "dimensional ESG breakdowns for stakeholder reporting."
            ),
            backstory=(
                "You are a quantitative ESG analyst who specializes in building "
                "hybrid risk assessment frameworks that integrate structured data "
                "models with NLP-based sentiment and topic analysis. Your fusion "
                "methodology is used by major pension funds to assess portfolio "
                "ESG risk across thousands of companies simultaneously."
            ),
            verbose=settings.VERBOSE_AGENTS,
            allow_delegation=False,
        )

    def fuse(
        self,
        ml_result: Dict[str, Any],
        zero_shot_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fuse ML and NLP results into a composite ESG score.

        Args:
            ml_result: Output from MLRiskModelerAgent.
            zero_shot_result: Output from ZeroShotAnalyzerAgent.

        Returns:
            Dict with composite_score, risk_tier, dimensional scores, confidence.
        """
        logger.info("Running hybrid fusion of ML and NLP ESG scores")

        # Extract ML score (0–100)
        ml_score = float(ml_result.get("risk_score_ml", 50.0))
        ml_confidence = float(ml_result.get("prediction_confidence", 0.75))

        # Extract NLP scores (0–1 → scale to 0–100)
        nlp_agg = zero_shot_result.get("aggregate_scores", {})
        nlp_env = float(nlp_agg.get("environmental", 0.3)) * 100
        nlp_social = float(nlp_agg.get("social", 0.3)) * 100
        nlp_gov = float(nlp_agg.get("governance", 0.3)) * 100
        nlp_overall = float(nlp_agg.get("overall_nlp_risk", 0.3)) * 100

        # If zero-shot analysis returned no data, use ML score
        nlp_analysis_complete = zero_shot_result.get("analysis_complete", False)
        if not nlp_analysis_complete or nlp_overall == 0:
            effective_nlp_weight = 0.0
            effective_ml_weight = 1.0
        else:
            effective_ml_weight = self.ml_weight
            effective_nlp_weight = self.nlp_weight

        # Composite score
        composite_score = (
            effective_ml_weight * ml_score +
            effective_nlp_weight * nlp_overall
        )
        composite_score = float(max(0.0, min(100.0, composite_score)))

        # Dimensional ESG scores
        dimensional_scores = self._compute_dimensional_scores(
            ml_result=ml_result,
            nlp_env=nlp_env,
            nlp_social=nlp_social,
            nlp_gov=nlp_gov,
            ml_weight=effective_ml_weight,
            nlp_weight=effective_nlp_weight,
        )

        # Risk tier
        risk_tier = self._assign_tier(composite_score)

        # Confidence interval
        confidence_interval = self._compute_confidence_interval(
            composite_score, ml_confidence, nlp_analysis_complete
        )

        # Emerging risks from NLP
        emerging_risks = zero_shot_result.get("emerging_risks", [])

        # Fusion metadata
        fusion_metadata = {
            "ml_weight_used": round(effective_ml_weight, 4),
            "nlp_weight_used": round(effective_nlp_weight, 4),
            "ml_score_input": round(ml_score, 2),
            "nlp_score_input": round(nlp_overall, 2),
            "nlp_analysis_complete": nlp_analysis_complete,
        }

        # Risk signals
        risk_signals = self._extract_risk_signals(
            ml_result, zero_shot_result, composite_score
        )

        result = {
            "composite_esg_score": round(composite_score, 2),
            "risk_tier": risk_tier,
            "risk_tier_description": self._tier_description(risk_tier),
            "dimensional_scores": dimensional_scores,
            "confidence_interval": confidence_interval,
            "emerging_risks": emerging_risks[:5],
            "risk_signals": risk_signals,
            "fusion_metadata": fusion_metadata,
            "investment_recommendation": self._generate_recommendation(
                risk_tier, composite_score, emerging_risks
            ),
        }

        logger.info(
            f"Fusion complete: composite_score={composite_score:.2f}, tier={risk_tier}"
        )
        return result

    def _compute_dimensional_scores(
        self,
        ml_result: Dict[str, Any],
        nlp_env: float,
        nlp_social: float,
        nlp_gov: float,
        ml_weight: float,
        nlp_weight: float,
    ) -> Dict[str, Any]:
        """Compute E, S, G dimensional scores."""
        # Derive ML dimensional scores from feature importances
        importances = ml_result.get("feature_importances", {})
        ml_score = float(ml_result.get("risk_score_ml", 50.0))

        env_features = ["carbon_emissions", "water_usage", "renewable_energy_pct"]
        social_features = ["employee_turnover", "board_diversity", "supply_chain_risk"]
        gov_features = ["controversy_score"]

        def weighted_ml_dim(features):
            total_imp = sum(importances.get(f, 0) for f in features)
            return ml_score * total_imp if total_imp > 0 else ml_score / 3

        ml_env = weighted_ml_dim(env_features)
        ml_social = weighted_ml_dim(social_features)
        ml_gov = weighted_ml_dim(gov_features)

        return {
            "environmental": {
                "score": round(ml_weight * ml_env + nlp_weight * nlp_env, 2),
                "ml_component": round(ml_env, 2),
                "nlp_component": round(nlp_env, 2),
            },
            "social": {
                "score": round(ml_weight * ml_social + nlp_weight * nlp_social, 2),
                "ml_component": round(ml_social, 2),
                "nlp_component": round(nlp_social, 2),
            },
            "governance": {
                "score": round(ml_weight * ml_gov + nlp_weight * nlp_gov, 2),
                "ml_component": round(ml_gov, 2),
                "nlp_component": round(nlp_gov, 2),
            },
        }

    def _assign_tier(self, score: float) -> str:
        """Assign risk tier from composite score."""
        for tier, (low, high) in RISK_TIER_THRESHOLDS.items():
            if low <= score < high:
                return tier
        return "CRITICAL"

    def _tier_description(self, tier: str) -> str:
        descriptions = {
            "LOW": "Company demonstrates strong ESG practices with minimal risk exposure.",
            "MEDIUM": "Moderate ESG risk with some areas requiring improvement.",
            "HIGH": "Significant ESG risk requiring immediate management attention.",
            "CRITICAL": "Severe ESG risk posing material financial and reputational threats.",
        }
        return descriptions.get(tier, "Unknown risk tier.")

    def _compute_confidence_interval(
        self,
        score: float,
        ml_confidence: float,
        nlp_complete: bool,
    ) -> Dict[str, float]:
        """Compute confidence interval around composite score."""
        base_uncertainty = (1 - ml_confidence) * 15
        if not nlp_complete:
            base_uncertainty += 10
        return {
            "lower": round(max(0, score - base_uncertainty), 2),
            "upper": round(min(100, score + base_uncertainty), 2),
            "uncertainty": round(base_uncertainty, 2),
        }

    def _extract_risk_signals(
        self,
        ml_result: Dict[str, Any],
        zero_shot_result: Dict[str, Any],
        composite_score: float,
    ) -> list:
        """Extract key risk signals from both sources."""
        signals = []

        # ML-based signals
        probs = ml_result.get("class_probabilities", {})
        if probs.get("high", 0) > 0.5:
            signals.append({
                "type": "ML_HIGH_RISK",
                "message": f"ML model predicts HIGH risk with {probs['high']:.0%} probability",
                "severity": "high",
            })

        importances = ml_result.get("feature_importances", {})
        top_feature = next(iter(importances), None)
        if top_feature:
            signals.append({
                "type": "TOP_RISK_DRIVER",
                "message": f"Primary risk driver: {top_feature} (importance: {importances[top_feature]:.2f})",
                "severity": "medium",
            })

        # NLP-based signals
        for risk in zero_shot_result.get("emerging_risks", [])[:3]:
            signals.append({
                "type": "EMERGING_NLP_RISK",
                "message": f"Emerging risk detected: {risk['risk']} (confidence: {risk['confidence']:.0%})",
                "severity": "high" if risk["confidence"] > 0.7 else "medium",
            })

        # Score-based signals
        if composite_score >= 75:
            signals.append({
                "type": "CRITICAL_SCORE",
                "message": "Composite ESG score exceeds CRITICAL threshold (75+)",
                "severity": "critical",
            })

        return signals

    def _generate_recommendation(
        self,
        risk_tier: str,
        score: float,
        emerging_risks: list,
    ) -> str:
        """Generate investment recommendation based on ESG profile."""
        recommendations = {
            "LOW": (
                "ESG profile supports inclusion in sustainable investment portfolios. "
                "Monitor for emerging risks on a quarterly basis."
            ),
            "MEDIUM": (
                "Conditional inclusion recommended with active engagement. "
                "Require management action plans on identified risk areas."
            ),
            "HIGH": (
                "Restrict new positions pending ESG improvement. Engage management "
                "immediately on material risk factors. Set 6-month review deadline."
            ),
            "CRITICAL": (
                "EXCLUDE from ESG-screened portfolios. Immediate divestment "
                "consideration for existing positions. Regulatory escalation may apply."
            ),
        }
        base = recommendations.get(risk_tier, "Insufficient data for recommendation.")

        if emerging_risks:
            top_risk = emerging_risks[0].get("risk", "unspecified")
            base += f" Priority emerging risk: {top_risk}."

        return base