"""
Integration tests for the full ESG audit agent pipeline.
"""

import pytest


class TestOrchestratorPipeline:
    """Integration tests for end-to-end ESG audit pipeline."""

    def test_full_pipeline_completes(self, sample_esg_data):
        """Test that the full pipeline runs without error."""
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        result = orchestrator.run_audit_pipeline(
            company_name="Test Corp",
            esg_data=sample_esg_data,
        )

        assert result.get("status") == "completed"
        assert "ingestion" in result
        assert "ml_risk" in result
        assert "fusion" in result
        assert "validation" in result
        assert "report" in result

    def test_pipeline_produces_composite_score(self, sample_esg_data):
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        result = orchestrator.run_audit_pipeline(
            company_name="Score Test Corp",
            esg_data=sample_esg_data,
        )

        composite = result["fusion"].get("composite_esg_score")
        assert composite is not None
        assert 0 <= composite <= 100

    def test_pipeline_assigns_risk_tier(self, sample_esg_data):
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        result = orchestrator.run_audit_pipeline(
            company_name="Tier Test Corp",
            esg_data=sample_esg_data,
        )

        tier = result["fusion"].get("risk_tier")
        assert tier in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_pipeline_generates_report_files(self, sample_esg_data, tmp_path):
        import os
        from agents.orchestrator import OrchestratorAgent
        from core.config import settings

        # Redirect reports to tmp_path
        original_dir = settings.REPORTS_DIR
        settings.__dict__["REPORTS_DIR"] = str(tmp_path)

        try:
            orchestrator = OrchestratorAgent()
            result = orchestrator.run_audit_pipeline(
                company_name="Report Test Corp",
                esg_data=sample_esg_data,
            )

            report = result.get("report", {})
            json_path = report.get("json_report_path")
            md_path = report.get("markdown_report_path")

            if json_path:
                assert os.path.exists(json_path), "JSON report should exist on disk"
            if md_path:
                assert os.path.exists(md_path), "Markdown report should exist on disk"
        finally:
            settings.__dict__["REPORTS_DIR"] = original_dir

    def test_pipeline_validation_runs(self, sample_esg_data):
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        result = orchestrator.run_audit_pipeline(
            company_name="Validation Corp",
            esg_data=sample_esg_data,
        )

        validation = result.get("validation", {})
        assert "validation_status" in validation
        assert validation["validation_status"] in {"PASS", "WARNING", "FAIL"}

    def test_pipeline_stages_recorded(self, sample_esg_data):
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        result = orchestrator.run_audit_pipeline(
            company_name="Stages Corp",
            esg_data=sample_esg_data,
        )

        stages = result.get("pipeline_stages", [])
        expected_stages = [
            "data_ingestion", "zero_shot_analysis", "ml_risk_modeling",
            "hybrid_fusion", "validation_explainability", "report_generation",
        ]
        for stage in expected_stages:
            assert stage in stages, f"Stage '{stage}' not recorded"

    def test_high_risk_company_gets_high_tier(self, high_risk_esg_data):
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        result = orchestrator.run_audit_pipeline(
            company_name="High Risk Corp",
            esg_data=high_risk_esg_data,
        )

        tier = result["fusion"].get("risk_tier")
        assert tier in {"HIGH", "CRITICAL"}, \
            f"High-risk data should produce HIGH or CRITICAL tier, got {tier}"