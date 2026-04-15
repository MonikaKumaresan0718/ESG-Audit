"""
Unit tests for ZeroShotAnalyzerAgent NLP classification.
"""

import pytest


class TestZeroShotAnalyzerAgent:
    """Tests for ZeroShotAnalyzerAgent classification logic."""

    def test_analyze_returns_expected_keys(self, sample_texts):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        result = agent.analyze(texts=sample_texts, company_name="Test Corp")

        required_keys = [
            "company_name", "texts_analyzed", "aggregate_scores",
            "emerging_risks", "model_used", "analysis_complete",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_analyze_empty_texts(self):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        result = agent.analyze(texts=[], company_name="Test Corp")

        assert result["texts_analyzed"] == 0
        assert result["analysis_complete"] is False
        assert result["aggregate_scores"]["environmental"] == 0.0

    def test_aggregate_scores_bounded(self, sample_texts):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        result = agent.analyze(texts=sample_texts[:2], company_name="Test Corp")

        agg = result["aggregate_scores"]
        for dim in ["environmental", "social", "governance", "overall_nlp_risk"]:
            if dim in agg:
                score = agg[dim]
                assert 0 <= score <= 1, f"{dim} score {score} out of [0, 1]"

    def test_company_name_in_result(self, sample_texts):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        result = agent.analyze(texts=sample_texts[:1], company_name="Acme Corp")

        assert result["company_name"] == "Acme Corp"

    def test_keyword_fallback_returns_scores(self):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        text = "The company has high carbon emissions and climate change risk."
        scores = agent._keyword_fallback(text)

        assert isinstance(scores, dict)
        assert len(scores) > 0
        assert "climate change" in scores
        assert scores["climate change"] > 0

    def test_keyword_fallback_all_scores_bounded(self, sample_texts):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        for text in sample_texts:
            scores = agent._keyword_fallback(text)
            for label, score in scores.items():
                assert 0 <= score <= 1, f"Score for '{label}' = {score} out of [0, 1]"

    def test_empty_result_structure(self):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent

        agent = ZeroShotAnalyzerAgent()
        result = agent._empty_result("Empty Corp")

        assert result["company_name"] == "Empty Corp"
        assert result["texts_analyzed"] == 0
        assert result["analysis_complete"] is False

    def test_texts_analyzed_count(self, sample_texts):
        from agents.zero_shot_analyzer import ZeroShotAnalyzerAgent
        from core.config import settings

        agent = ZeroShotAnalyzerAgent()
        limit = min(len(sample_texts), settings.MAX_TEXTS_FOR_ZERO_SHOT)
        result = agent.analyze(texts=sample_texts, company_name="Count Test")

        assert result["texts_analyzed"] <= limit