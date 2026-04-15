"""
Unit tests for MLRiskModelerAgent and ML feature engineering pipeline.
"""

import pytest
import numpy as np
import pandas as pd


class TestESGFeatureEngineer:
    """Tests for ESGFeatureEngineer feature transformation."""

    def test_transform_basic_dict(self, sample_esg_data):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform(sample_esg_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.shape[1] > 7, "Should have more columns than raw inputs after engineering"

    def test_transform_adds_interaction_features(self, sample_esg_data):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform(sample_esg_data)

        assert "env_risk_composite" in result.columns
        assert "social_risk_composite" in result.columns
        assert "gov_risk_composite" in result.columns
        assert "carbon_adjusted" in result.columns

    def test_transform_adds_risk_proxies(self, sample_esg_data):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform(sample_esg_data)

        assert "env_risk_proxy" in result.columns
        assert "social_risk_proxy" in result.columns
        assert "gov_risk_proxy" in result.columns
        assert "overall_risk_proxy" in result.columns

    def test_risk_proxies_bounded_0_1(self, sample_esg_data):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform(sample_esg_data)

        for col in ["env_risk_proxy", "social_risk_proxy", "gov_risk_proxy", "overall_risk_proxy"]:
            assert 0 <= result[col].iloc[0] <= 1, f"{col} out of [0, 1] range"

    def test_transform_handles_missing_fields(self):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        # Partial data – should not raise
        partial_data = {"carbon_emissions": 300.0, "controversy_score": 5.0}
        result = engineer.transform(partial_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_transform_handles_empty_dict(self):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform({})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_high_risk_proxy_for_high_risk_data(self, high_risk_esg_data):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform(high_risk_esg_data)

        assert result["overall_risk_proxy"].iloc[0] > 0.5, \
            "High-risk data should produce overall_risk_proxy > 0.5"

    def test_low_risk_proxy_for_low_risk_data(self, low_risk_esg_data):
        from ml.feature_engineering import ESGFeatureEngineer

        engineer = ESGFeatureEngineer()
        result = engineer.transform(low_risk_esg_data)

        assert result["overall_risk_proxy"].iloc[0] < 0.5, \
            "Low-risk data should produce overall_risk_proxy < 0.5"


class TestMLRiskModelerAgent:
    """Tests for MLRiskModelerAgent prediction logic."""

    def test_predict_returns_expected_keys(self, sample_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict(sample_esg_data)

        required_keys = [
            "risk_score_ml", "risk_tier_ml", "predicted_class",
            "class_probabilities", "feature_importances",
            "model_version", "prediction_confidence",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_score_in_range(self, sample_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict(sample_esg_data)

        assert 0 <= result["risk_score_ml"] <= 100

    def test_predict_tier_is_valid(self, sample_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict(sample_esg_data)

        assert result["risk_tier_ml"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_high_risk_data_higher_score(self, high_risk_esg_data, low_risk_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        high_result = agent.predict(high_risk_esg_data)
        low_result = agent.predict(low_risk_esg_data)

        assert high_result["risk_score_ml"] > low_result["risk_score_ml"], \
            "High-risk data should produce higher ML score than low-risk data"

    def test_predict_handles_empty_data(self):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict({})

        assert "risk_score_ml" in result
        assert "risk_tier_ml" in result

    def test_class_probabilities_sum_to_one(self, sample_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict(sample_esg_data)

        probs = result["class_probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected 1.0"

    def test_feature_importances_sum_approximately_one(self, sample_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict(sample_esg_data)

        fi = result.get("feature_importances", {})
        if fi:
            total = sum(fi.values())
            assert 0.9 <= total <= 1.1, f"Feature importances sum to {total}"

    def test_risk_tier_consistency_with_score(self, sample_esg_data):
        from agents.ml_risk_modeler import MLRiskModelerAgent

        agent = MLRiskModelerAgent()
        result = agent.predict(sample_esg_data)

        score = result["risk_score_ml"]
        tier = result["risk_tier_ml"]

        if score < 25:
            assert tier == "LOW"
        elif score < 50:
            assert tier == "MEDIUM"
        elif score < 75:
            assert tier == "HIGH"
        else:
            assert tier == "CRITICAL"