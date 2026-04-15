"""
ReportGeneratorAgent – Produces comprehensive ESG audit reports in JSON and
Markdown formats, and optionally renders a PDF using Jinja2 templates.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from crewai import Agent

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class ReportGeneratorAgent:
    """
    Generates comprehensive, multi-format ESG audit reports.
    Supports JSON, Markdown, and optional PDF output.
    """

    def __init__(self):
        self.output_dir = settings.REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.agent = self._build_crewai_agent()

    def _build_crewai_agent(self) -> Agent:
        return Agent(
            role="ESG Audit Report Generator",
            goal=(
                "Produce comprehensive, professionally formatted ESG audit reports "
                "in JSON and Markdown formats with optional PDF rendering. Reports "
                "must be suitable for institutional investors, board presentations, "
                "and regulatory disclosure. Ensure executive summaries are concise "
                "and actionable."
            ),
            backstory=(
                "You are a senior ESG communications specialist and financial writer "
                "who has produced sustainability reports for major asset managers and "
                "public companies. You understand how to translate complex quantitative "
                "ESG data into clear narratives for diverse audiences. You are expert "
                "in GRI, SASB, and TCFD disclosure formats."
            ),
            verbose=settings.VERBOSE_AGENTS,
            allow_delegation=False,
        )

    def generate(
        self,
        audit_id: str,
        company_name: str,
        ingestion: Dict[str, Any],
        zero_shot: Dict[str, Any],
        ml_risk: Dict[str, Any],
        fusion: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate the full audit report in all supported formats.

        Returns:
            Dict with file paths and report summary.
        """
        logger.info(f"Generating ESG audit report for {company_name} [{audit_id}]")

        # Assemble report data
        report_data = self._assemble_report_data(
            audit_id=audit_id,
            company_name=company_name,
            ingestion=ingestion,
            zero_shot=zero_shot,
            ml_risk=ml_risk,
            fusion=fusion,
            validation=validation,
        )

        # Generate files
        json_path = self._write_json_report(audit_id, report_data)
        md_path = self._write_markdown_report(audit_id, report_data)
        pdf_path = None

        if settings.ENABLE_PDF_REPORTS:
            try:
                pdf_path = self._write_pdf_report(audit_id, report_data)
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")

        return {
            "audit_id": audit_id,
            "company_name": company_name,
            "report_timestamp": report_data["generated_at"],
            "json_report_path": json_path,
            "markdown_report_path": md_path,
            "pdf_report_path": pdf_path,
            "executive_summary": report_data["executive_summary"],
            "composite_score": fusion.get("composite_esg_score"),
            "risk_tier": fusion.get("risk_tier"),
            "total_pages": self._estimate_pages(report_data),
        }

    def _assemble_report_data(
        self,
        audit_id: str,
        company_name: str,
        ingestion: Dict[str, Any],
        zero_shot: Dict[str, Any],
        ml_risk: Dict[str, Any],
        fusion: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble all pipeline outputs into a unified report structure."""
        composite_score = fusion.get("composite_esg_score", 0)
        risk_tier = fusion.get("risk_tier", "UNKNOWN")

        return {
            "audit_id": audit_id,
            "company_name": company_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "report_version": "1.0.0",
            "executive_summary": self._generate_executive_summary(
                company_name, composite_score, risk_tier, fusion, validation
            ),
            "esg_scores": {
                "composite_score": composite_score,
                "risk_tier": risk_tier,
                "risk_tier_description": fusion.get("risk_tier_description", ""),
                "dimensional_scores": fusion.get("dimensional_scores", {}),
                "confidence_interval": fusion.get("confidence_interval", {}),
            },
            "ml_analysis": {
                "risk_score": ml_risk.get("risk_score_ml"),
                "risk_tier": ml_risk.get("risk_tier_ml"),
                "model_version": ml_risk.get("model_version"),
                "prediction_confidence": ml_risk.get("prediction_confidence"),
                "class_probabilities": ml_risk.get("class_probabilities", {}),
                "feature_importances": ml_risk.get("feature_importances", {}),
                "top_risk_drivers": list(ml_risk.get("feature_importances", {}).keys())[:5],
            },
            "nlp_analysis": {
                "model": zero_shot.get("model_used", "bart-large-mnli"),
                "texts_analyzed": zero_shot.get("texts_analyzed", 0),
                "aggregate_scores": zero_shot.get("aggregate_scores", {}),
                "emerging_risks": zero_shot.get("emerging_risks", []),
                "top_risk_labels": self._get_top_labels(zero_shot),
            },
            "data_sources": {
                "structured_data": ingestion.get("structured_data", {}),
                "text_sources": ingestion.get("text_sources", []),
                "embedding_count": ingestion.get("embedding_count", 0),
                "ingestion_errors": ingestion.get("errors", []),
            },
            "validation": {
                "status": validation.get("validation_status", "UNKNOWN"),
                "gri_flags": validation.get("regulatory_checks", {}).get("gri_flags", []),
                "sasb_flags": validation.get("regulatory_checks", {}).get("sasb_flags", []),
                "tcfd_flags": validation.get("regulatory_checks", {}).get("tcfd_flags", []),
                "consistency_flags": validation.get("consistency_flags", []),
                "total_flags": validation.get("total_flags", 0),
                "frameworks_checked": validation.get("frameworks_checked", []),
                "audit_trail": validation.get("audit_trail", {}),
            },
            "explainability": {
                "shap_values": validation.get("shap_values", {}),
                "lime_explanation": validation.get("lime_explanation", {}),
                "confidence_intervals": validation.get("confidence_intervals", {}),
            },
            "risk_signals": fusion.get("risk_signals", []),
            "investment_recommendation": fusion.get("investment_recommendation", ""),
            "fusion_metadata": fusion.get("fusion_metadata", {}),
            "disclaimer": (
                "This ESG audit report is generated by an automated AI system and "
                "is intended for informational purposes only. It does not constitute "
                "financial, legal, or investment advice. Results should be verified "
                "by qualified ESG analysts before making investment decisions."
            ),
        }

    def _generate_executive_summary(
        self,
        company_name: str,
        composite_score: float,
        risk_tier: str,
        fusion: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> str:
        """Generate a concise executive summary paragraph."""
        dim = fusion.get("dimensional_scores", {})
        env_score = dim.get("environmental", {}).get("score", 0)
        social_score = dim.get("social", {}).get("score", 0)
        gov_score = dim.get("governance", {}).get("score", 0)
        val_status = validation.get("validation_status", "UNKNOWN")
        emerging = fusion.get("emerging_risks", [])
        top_risk = emerging[0].get("risk", "N/A") if emerging else "None identified"

        tier_color = {
            "LOW": "strong ESG performance",
            "MEDIUM": "moderate ESG risk",
            "HIGH": "elevated ESG risk",
            "CRITICAL": "critical ESG risk",
        }.get(risk_tier, "unassessed ESG risk")

        return (
            f"{company_name} demonstrates {tier_color} with a composite ESG score of "
            f"{composite_score:.1f}/100 (Risk Tier: {risk_tier}). Dimensional analysis reveals "
            f"Environmental Score: {env_score:.1f}, Social Score: {social_score:.1f}, and "
            f"Governance Score: {gov_score:.1f}. Regulatory validation against GRI, SASB, and "
            f"TCFD frameworks returned status: {val_status}. Primary emerging risk identified: "
            f"{top_risk}. {fusion.get('investment_recommendation', '')}"
        )

    def _get_top_labels(self, zero_shot: Dict[str, Any]) -> list:
        """Extract top NLP risk labels."""
        labels = zero_shot.get("label_scores", {})
        return sorted(labels.items(), key=lambda x: x[1], reverse=True)[:5]

    def _write_json_report(self, audit_id: str, data: Dict[str, Any]) -> str:
        """Write JSON report to disk."""
        path = os.path.join(self.output_dir, f"esg_audit_{audit_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON report written to {path}")
        return path

    def _write_markdown_report(self, audit_id: str, data: Dict[str, Any]) -> str:
        """Write Markdown report to disk."""
        path = os.path.join(self.output_dir, f"esg_audit_{audit_id}.md")
        md_content = self._render_markdown(data)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Markdown report written to {path}")
        return path

    def _render_markdown(self, data: Dict[str, Any]) -> str:
        """Render full Markdown report."""
        try:
            from tools.report_renderer import MarkdownRenderer
            renderer = MarkdownRenderer()
            return renderer.render(data)
        except Exception:
            return self._fallback_markdown(data)

    def _fallback_markdown(self, data: Dict[str, Any]) -> str:
        """Fallback Markdown rendering without Jinja2."""
        scores = data.get("esg_scores", {})
        dim = scores.get("dimensional_scores", {})
        ml = data.get("ml_analysis", {})
        nlp = data.get("nlp_analysis", {})
        val = data.get("validation", {})

        lines = [
            f"# ESG Audit Report: {data['company_name']}",
            f"",
            f"**Audit ID:** {data['audit_id']}  ",
            f"**Generated:** {data['generated_at']}  ",
            f"**Report Version:** {data['report_version']}",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            data.get("executive_summary", ""),
            f"",
            f"---",
            f"",
            f"## ESG Score Overview",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Composite ESG Score** | {scores.get('composite_score', 'N/A')}/100 |",
            f"| **Risk Tier** | {scores.get('risk_tier', 'N/A')} |",
            f"| Environmental Score | {dim.get('environmental', {}).get('score', 'N/A')} |",
            f"| Social Score | {dim.get('social', {}).get('score', 'N/A')} |",
            f"| Governance Score | {dim.get('governance', {}).get('score', 'N/A')} |",
            f"",
            f"---",
            f"",
            f"## ML Risk Analysis",
            f"",
            f"- **ML Risk Score:** {ml.get('risk_score', 'N/A')}/100",
            f"- **Risk Tier:** {ml.get('risk_tier', 'N/A')}",
            f"- **Model Version:** {ml.get('model_version', 'N/A')}",
            f"- **Prediction Confidence:** {ml.get('prediction_confidence', 'N/A')}",
            f"",
            f"### Top Risk Drivers",
            f"",
        ]

        for feature, importance in list(
            ml.get("feature_importances", {}).items()
        )[:5]:
            lines.append(f"- `{feature}`: {importance:.4f}")

        lines += [
            f"",
            f"---",
            f"",
            f"## Zero-Shot NLP Analysis",
            f"",
            f"- **Model:** {nlp.get('model', 'N/A')}",
            f"- **Texts Analyzed:** {nlp.get('texts_analyzed', 'N/A')}",
            f"",
            f"### Aggregate NLP Scores",
            f"",
        ]

        agg = nlp.get("aggregate_scores", {})
        for dim_name, score in agg.items():
            lines.append(f"- **{dim_name.capitalize()}:** {score:.4f}")

        lines += [
            f"",
            f"### Emerging Risks Detected",
            f"",
        ]

        for risk in nlp.get("emerging_risks", [])[:5]:
            lines.append(
                f"- **{risk.get('risk', 'Unknown')}** "
                f"(confidence: {risk.get('confidence', 0):.2%})"
            )

        lines += [
            f"",
            f"---",
            f"",
            f"## Regulatory Validation",
            f"",
            f"**Validation Status:** {val.get('status', 'N/A')}  ",
            f"**Frameworks Checked:** {', '.join(val.get('frameworks_checked', []))}  ",
            f"**Total Flags:** {val.get('total_flags', 0)}",
            f"",
            f"### GRI Flags",
        ]

        for flag in val.get("gri_flags", []):
            lines.append(f"- [{flag.get('severity', '').upper()}] {flag.get('message', '')}")

        lines += [
            f"",
            f"### SASB Flags",
        ]
        for flag in val.get("sasb_flags", []):
            lines.append(f"- [{flag.get('severity', '').upper()}] {flag.get('message', '')}")

        lines += [
            f"",
            f"### TCFD Flags",
        ]
        for flag in val.get("tcfd_flags", []):
            lines.append(f"- [{flag.get('severity', '').upper()}] {flag.get('message', '')}")

        lines += [
            f"",
            f"---",
            f"",
            f"## Investment Recommendation",
            f"",
            data.get("investment_recommendation", "No recommendation available."),
            f"",
            f"---",
            f"",
            f"## Disclaimer",
            f"",
            f"*{data.get('disclaimer', '')}*",
            f"",
        ]

        return "\n".join(lines)

    def _write_pdf_report(self, audit_id: str, data: Dict[str, Any]) -> str:
        """Render and write PDF report using Jinja2 + WeasyPrint."""
        from tools.report_renderer import PDFRenderer
        path = os.path.join(self.output_dir, f"esg_audit_{audit_id}.pdf")
        renderer = PDFRenderer(template_dir=settings.TEMPLATES_DIR)
        renderer.render_to_file(data, output_path=path)
        logger.info(f"PDF report written to {path}")
        return path

    def _estimate_pages(self, data: Dict[str, Any]) -> int:
        """Estimate report page count."""
        base_pages = 5
        flag_pages = len(data.get("validation", {}).get("gri_flags", [])) // 5
        return base_pages + flag_pages