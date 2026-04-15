"""
ZeroShotAnalyzerAgent – Uses Hugging Face's facebook/bart-large-mnli for zero-shot
classification into ESG risk categories with structured JSON output.
"""

import json
from typing import Any, Dict, List, Optional

from crewai import Agent

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

ESG_CANDIDATE_LABELS = [
    "environmental risk",
    "social risk",
    "governance risk",
    "climate change",
    "carbon emissions",
    "water scarcity",
    "biodiversity loss",
    "supply chain risk",
    "labor rights violation",
    "board diversity",
    "executive compensation",
    "data privacy",
    "regulatory compliance",
    "greenwashing",
    "human rights",
    "community impact",
]

EMERGING_RISK_LABELS = [
    "emerging environmental risk",
    "regulatory non-compliance",
    "reputational risk",
    "financial materiality",
    "systemic ESG failure",
    "transition risk",
    "physical climate risk",
    "social inequality",
    "corruption and bribery",
]


class ZeroShotAnalyzerAgent:
    """
    Performs zero-shot ESG risk classification using HuggingFace BART-large-MNLI.
    Returns structured JSON with confidence scores and emerging risks.
    """

    def __init__(self):
        self._classifier = None
        self.agent = self._build_crewai_agent()

    def _build_crewai_agent(self) -> Agent:
        return Agent(
            role="Zero-Shot ESG Risk Analyzer",
            goal=(
                "Use zero-shot NLP classification to identify and score Environmental, "
                "Social, and Governance risks in unstructured sustainability texts "
                "without requiring fine-tuned training data. Detect emerging risks "
                "with high confidence and return structured JSON results."
            ),
            backstory=(
                "You are an NLP research scientist specializing in zero-shot learning "
                "applied to ESG analysis. You have expertise in transformer models and "
                "have developed classification pipelines for major ESG rating agencies. "
                "You understand sustainability frameworks and can identify subtle risk "
                "signals in corporate disclosures."
            ),
            verbose=settings.VERBOSE_AGENTS,
            allow_delegation=False,
        )

    @property
    def classifier(self):
        """Lazy-load the zero-shot classification pipeline."""
        if self._classifier is None:
            try:
                from transformers import pipeline

                logger.info(f"Loading zero-shot model: {settings.ZERO_SHOT_MODEL}")
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model=settings.ZERO_SHOT_MODEL,
                    device=settings.DEVICE,
                )
                logger.info("Zero-shot classifier loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load zero-shot model: {e}")
                self._classifier = None
        return self._classifier

    def analyze(
        self,
        texts: List[str],
        company_name: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze texts using zero-shot classification.

        Args:
            texts: List of text chunks to analyze.
            company_name: Company name for metadata.
            top_k: Number of top labels to return per text.

        Returns:
            Structured dict with ESG category scores and emerging risks.
        """
        logger.info(
            f"Running zero-shot analysis for {company_name} on {len(texts)} texts"
        )

        if not texts:
            return self._empty_result(company_name)

        # Aggregate scores per ESG dimension
        env_scores: List[float] = []
        social_scores: List[float] = []
        gov_scores: List[float] = []
        all_label_scores: Dict[str, List[float]] = {
            label: [] for label in ESG_CANDIDATE_LABELS
        }
        emerging_risks: List[Dict[str, Any]] = []
        text_results: List[Dict[str, Any]] = []

        for i, text in enumerate(texts[:settings.MAX_TEXTS_FOR_ZERO_SHOT]):
            try:
                if self.classifier is not None:
                    result = self.classifier(
                        text[:512],  # BART token limit
                        candidate_labels=ESG_CANDIDATE_LABELS,
                        multi_label=True,
                    )
                    label_scores = dict(
                        zip(result["labels"], result["scores"])
                    )
                else:
                    # Fallback: keyword-based scoring
                    label_scores = self._keyword_fallback(text)

                # Aggregate by dimension
                env_score = max(
                    label_scores.get("environmental risk", 0),
                    label_scores.get("climate change", 0),
                    label_scores.get("carbon emissions", 0),
                    label_scores.get("water scarcity", 0),
                    label_scores.get("biodiversity loss", 0),
                )
                social_score = max(
                    label_scores.get("social risk", 0),
                    label_scores.get("labor rights violation", 0),
                    label_scores.get("human rights", 0),
                    label_scores.get("community impact", 0),
                    label_scores.get("supply chain risk", 0),
                )
                gov_score = max(
                    label_scores.get("governance risk", 0),
                    label_scores.get("board diversity", 0),
                    label_scores.get("executive compensation", 0),
                    label_scores.get("data privacy", 0),
                    label_scores.get("regulatory compliance", 0),
                )

                env_scores.append(env_score)
                social_scores.append(social_score)
                gov_scores.append(gov_score)

                for label, score in label_scores.items():
                    all_label_scores[label].append(score)

                # Detect emerging risks
                if self.classifier is not None:
                    emerging_result = self.classifier(
                        text[:512],
                        candidate_labels=EMERGING_RISK_LABELS,
                        multi_label=True,
                    )
                    for label, score in zip(
                        emerging_result["labels"], emerging_result["scores"]
                    ):
                        if score > settings.EMERGING_RISK_THRESHOLD:
                            emerging_risks.append(
                                {
                                    "risk": label,
                                    "confidence": round(float(score), 4),
                                    "source_text_index": i,
                                    "text_excerpt": text[:200],
                                }
                            )

                text_results.append(
                    {
                        "text_index": i,
                        "environmental_score": round(env_score, 4),
                        "social_score": round(social_score, 4),
                        "governance_score": round(gov_score, 4),
                        "top_labels": sorted(
                            label_scores.items(), key=lambda x: x[1], reverse=True
                        )[:top_k],
                    }
                )

            except Exception as e:
                logger.warning(f"Zero-shot analysis failed for text {i}: {e}")
                text_results.append({"text_index": i, "error": str(e)})

        # Compute aggregate scores
        agg_env = float(sum(env_scores) / len(env_scores)) if env_scores else 0.0
        agg_social = float(sum(social_scores) / len(social_scores)) if social_scores else 0.0
        agg_gov = float(sum(gov_scores) / len(gov_scores)) if gov_scores else 0.0

        # Deduplicate emerging risks
        seen_risks = set()
        unique_risks = []
        for risk in sorted(emerging_risks, key=lambda x: x["confidence"], reverse=True):
            if risk["risk"] not in seen_risks:
                seen_risks.add(risk["risk"])
                unique_risks.append(risk)

        # Average label scores
        avg_label_scores = {
            label: round(float(sum(scores) / len(scores)), 4)
            if scores
            else 0.0
            for label, scores in all_label_scores.items()
        }

        return {
            "company_name": company_name,
            "texts_analyzed": len(text_results),
            "aggregate_scores": {
                "environmental": round(agg_env, 4),
                "social": round(agg_social, 4),
                "governance": round(agg_gov, 4),
                "overall_nlp_risk": round((agg_env + agg_social + agg_gov) / 3, 4),
            },
            "label_scores": avg_label_scores,
            "emerging_risks": unique_risks[:10],
            "text_level_results": text_results,
            "model_used": settings.ZERO_SHOT_MODEL,
            "analysis_complete": True,
        }

    def _keyword_fallback(self, text: str) -> Dict[str, float]:
        """Simple keyword-based scoring as fallback when model unavailable."""
        text_lower = text.lower()
        scores = {}

        keyword_map = {
            "environmental risk": ["pollution", "emissions", "waste", "toxic", "contamination"],
            "social risk": ["labor", "workers", "discrimination", "harassment", "inequality"],
            "governance risk": ["corruption", "fraud", "misconduct", "violation", "breach"],
            "climate change": ["climate", "warming", "greenhouse", "carbon", "co2"],
            "carbon emissions": ["carbon", "co2", "emissions", "greenhouse", "scope"],
            "water scarcity": ["water", "drought", "aquifer", "freshwater"],
            "biodiversity loss": ["biodiversity", "species", "ecosystem", "habitat", "deforestation"],
            "supply chain risk": ["supply chain", "supplier", "vendor", "procurement"],
            "labor rights violation": ["labor", "workers rights", "child labor", "forced labor"],
            "board diversity": ["board", "diversity", "gender", "independence", "directors"],
            "executive compensation": ["compensation", "salary", "bonus", "incentive", "pay"],
            "data privacy": ["privacy", "data", "gdpr", "cybersecurity", "breach"],
            "regulatory compliance": ["compliance", "regulation", "penalty", "fine", "violation"],
            "greenwashing": ["greenwashing", "misleading", "false claims", "exaggerated"],
            "human rights": ["human rights", "forced labor", "trafficking", "abuse"],
            "community impact": ["community", "local", "indigenous", "displacement"],
        }

        for label, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[label] = min(count * 0.15, 0.9)

        return scores

    def _empty_result(self, company_name: str) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            "company_name": company_name,
            "texts_analyzed": 0,
            "aggregate_scores": {
                "environmental": 0.0,
                "social": 0.0,
                "governance": 0.0,
                "overall_nlp_risk": 0.0,
            },
            "label_scores": {},
            "emerging_risks": [],
            "text_level_results": [],
            "model_used": settings.ZERO_SHOT_MODEL,
            "analysis_complete": False,
        }