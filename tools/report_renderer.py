"""
Jinja2 + WeasyPrint report renderer for ESG audit PDF and Markdown generation.
"""

import os
from typing import Any, Dict, Optional

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class MarkdownRenderer:
    """Render ESG audit reports in Markdown using Jinja2 templates."""

    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or settings.TEMPLATES_DIR

    def render(self, data: Dict[str, Any]) -> str:
        """
        Render Markdown report using Jinja2 template.

        Args:
            data: Report data dict.

        Returns:
            Rendered Markdown string.
        """
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape

            env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )

            template = env.get_template("executive_summary.md.j2")
            return template.render(**data)
        except Exception as e:
            logger.warning(f"Jinja2 Markdown rendering failed: {e}")
            return self._fallback_render(data)

    def _fallback_render(self, data: Dict[str, Any]) -> str:
        """Simple fallback Markdown rendering."""
        company = data.get("company_name", "Unknown")
        audit_id = data.get("audit_id", "N/A")
        score = data.get("esg_scores", {}).get("composite_score", "N/A")
        tier = data.get("esg_scores", {}).get("risk_tier", "N/A")

        return f"""# ESG Audit Report: {company}

**Audit ID:** {audit_id}
**Generated:** {data.get('generated_at', 'N/A')}

## Executive Summary

{data.get('executive_summary', 'No summary available.')}

## Key Results

- **Composite ESG Score:** {score}/100
- **Risk Tier:** {tier}
- **Validation Status:** {data.get('validation', {}).get('status', 'N/A')}

## Investment Recommendation

{data.get('investment_recommendation', 'No recommendation available.')}

---
*{data.get('disclaimer', '')}*
"""


class PDFRenderer:
    """Render ESG audit reports as PDF using Jinja2 HTML + WeasyPrint."""

    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or settings.TEMPLATES_DIR

    def render_to_string(self, data: Dict[str, Any]) -> str:
        """Render HTML string from Jinja2 template."""
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Register custom filters
        env.filters["percentage"] = lambda v: f"{float(v):.1%}"
        env.filters["round2"] = lambda v: f"{float(v):.2f}"
        env.filters["tier_color"] = self._tier_color

        template = env.get_template("esg_report.html.j2")
        return template.render(**data)

    def render_to_file(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Render report to PDF file.

        Args:
            data: Report data dict.
            output_path: Output PDF file path.

        Returns:
            Path to generated PDF.
        """
        try:
            from weasyprint import HTML

            html_content = self.render_to_string(data)
            HTML(string=html_content).write_pdf(output_path)
            logger.info(f"PDF report generated: {output_path}")
            return output_path
        except ImportError:
            logger.warning("WeasyPrint not installed; saving HTML instead")
            html_path = output_path.replace(".pdf", ".html")
            try:
                html_content = self.render_to_string(data)
            except Exception:
                html_content = self._fallback_html(data)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return html_path
        except Exception as e:
            logger.error(f"PDF rendering failed: {e}")
            raise

    def _tier_color(self, tier: str) -> str:
        """Return CSS color class for risk tier."""
        colors = {
            "LOW": "#27ae60",
            "MEDIUM": "#f39c12",
            "HIGH": "#e74c3c",
            "CRITICAL": "#8e44ad",
        }
        return colors.get(tier, "#95a5a6")

    def _fallback_html(self, data: Dict[str, Any]) -> str:
        """Simple HTML fallback."""
        company = data.get("company_name", "Unknown")
        score = data.get("esg_scores", {}).get("composite_score", "N/A")
        tier = data.get("esg_scores", {}).get("risk_tier", "N/A")

        return f"""<!DOCTYPE html>
<html>
<head><title>ESG Audit Report: {company}</title></head>
<body>
<h1>ESG Audit Report: {company}</h1>
<p><strong>Audit ID:</strong> {data.get('audit_id', 'N/A')}</p>
<p><strong>Generated:</strong> {data.get('generated_at', 'N/A')}</p>
<h2>Executive Summary</h2>
<p>{data.get('executive_summary', 'N/A')}</p>
<h2>Key Results</h2>
<ul>
  <li><strong>Composite ESG Score:</strong> {score}/100</li>
  <li><strong>Risk Tier:</strong> {tier}</li>
</ul>
<h2>Recommendation</h2>
<p>{data.get('investment_recommendation', 'N/A')}</p>
<hr/>
<p><em>{data.get('disclaimer', '')}</em></p>
</body>
</html>"""