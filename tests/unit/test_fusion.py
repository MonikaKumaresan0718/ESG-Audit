"""
Unit tests for HybridFusionAgent score combination logic.
"""

import pytest


class TestHybridFusionAgent:
    """Tests for HybridFusionAgent weighted ensemble."""

    def test_fuse_returns_expected_keys(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        result = agent.fuse(
            ml_result=mock_ml_result,
            zero_shot_result=mock_zero_shot_result,
        )

        required_keys = [
            "composite_esg_score", "risk_tier", "dimensional_scores",
            "confidence_interval", "fusion_metadata",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_composite_score_bounded(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        result = agent.fuse(mock_ml_result, mock_zero_shot_result)

        score = result["composite_esg_score"]
        assert 0 <= score <= 100, f"Composite score {score} out of [0, 100]"

    def test_risk_tier_is_valid(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        result = agent.fuse(mock_ml_result, mock_zero_shot_result)

        assert result["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_custom_weights(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        # Equal weights
        agent = HybridFusionAgent(ml_weight=0.5, nlp_weight=0.5)
        result = agent.fuse(mock_ml_result, mock_zero_shot_result)

        assert result["fusion_metadata"]["ml_weight_used"] == 0.5
        assert result["fusion_metadata"]["nlp_weight_used"] == 0.5

    def test_weights_invalid_raises(self):
        from agents.hybrid_fusion import HybridFusionAgent

        with pytest.raises(AssertionError):
            HybridFusionAgent(ml_weight=0.7, nlp_weight=0.7)

    def test_ml_only_when_nlp_incomplete(self, mock_ml_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        incomplete_nlp = {
            "aggregate_scores": {"environmental": 0, "social": 0, "governance": 0, "overall_nlp_risk": 0},
            "analysis_complete": False,
            "emerging_risks": [],
        }
        result = agent.fuse(mock_ml_result, incomplete_nlp)

        # Should fall back to ML-only weighting
        assert result["fusion_metadata"]["nlp_weight_used"] == 0.0
        assert result["fusion_metadata"]["ml_weight_used"] == 1.0

    def test_confidence_interval_structure(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        result = agent.fuse(mock_ml_result, mock_zero_shot_result)

        ci = result["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] <= result["composite_esg_score"] <= ci["upper"]

    def test_tier_assignment_low(self):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        assert agent._assign_tier(10) == "LOW"
        assert agent._assign_tier(24.9) == "LOW"

    def test_tier_assignment_medium(self):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        assert agent._assign_tier(25) == "MEDIUM"
        assert agent._assign_tier(49.9) == "MEDIUM"

    def test_tier_assignment_high(self):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        assert agent._assign_tier(50) == "HIGH"
        assert agent._assign_tier(74.9) == "HIGH"

    def test_tier_assignment_critical(self):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        assert agent._assign_tier(75) == "CRITICAL"
        assert agent._assign_tier(100) == "CRITICAL"

    def test_investment_recommendation_not_empty(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        result = agent.fuse(mock_ml_result, mock_zero_shot_result)

        rec = result.get("investment_recommendation", "")
        assert len(rec) > 20, "Investment recommendation should be substantive"

    def test_dimensional_scores_structure(self, mock_ml_result, mock_zero_shot_result):
        from agents.hybrid_fusion import HybridFusionAgent

        agent = HybridFusionAgent()
        result = agent.fuse(mock_ml_result, mock_zero_shot_result)

        dim = result["dimensional_scores"]
        for dimension in ["environmental", "social", "governance"]:
            assert dimension in dim
            assert "score" in dim[dimension]