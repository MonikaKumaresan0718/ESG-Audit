"""
Streamlit Report Viewer Page – View and download ESG audit reports.
"""

import streamlit as st
import requests
from typing import Any, Dict, Optional


API_BASE = "http://localhost:8000/v1"


def render_report_viewer() -> None:
    """Render the ESG audit report viewer page."""
    st.title("📄 ESG Audit Report Viewer")
    st.markdown("Enter an Audit ID to view the full results and download reports.")

    # Pre-fill from session state
    default_id = st.session_state.get("last_audit_id", "")

    audit_id = st.text_input(
        "Audit ID",
        value=default_id,
        placeholder="e.g. 3f7a2b91-...",
        help="The audit_id returned when you submitted the audit",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        fetch_btn = st.button("🔍 Load Report", type="primary")

    if not (fetch_btn or (audit_id and audit_id != default_id)):
        if not audit_id:
            st.info("💡 Submit an audit from **Run Audit** first, then paste the Audit ID here.")
        return

    if not audit_id.strip():
        st.warning("Please enter an Audit ID.")
        return

    with st.spinner(f"Loading audit `{audit_id[:12]}...`"):
        result = _fetch_audit(audit_id.strip())

    if not result:
        return

    status = result.get("status", "unknown")

    if status == "pending":
        st.warning("⏳ Audit is still **PENDING** – queued but not yet started.")
        return
    elif status == "running":
        st.info("⚡ Audit is currently **RUNNING**. Refresh in a few seconds.")
        return
    elif status == "failed":
        st.error(f"❌ Audit **FAILED**: {result.get('error_message', 'Unknown error')}")
        return

    # ── Completed audit ────────────────────────────────────────────────────
    company = result.get("company_name", "Unknown")
    composite = result.get("composite_esg_score") or 0
    tier = result.get("risk_tier") or "UNKNOWN"

    st.success(f"✅ Audit completed for **{company}**")

    # Score visualization
    from ui.components.risk_gauge import render_risk_gauge, render_esg_score_metrics
    from ui.components.esg_heatmap import (
        render_feature_importance_chart,
        render_nlp_scores_radar,
        render_esg_heatmap,
        render_emerging_risks_table,
        render_validation_flags_summary,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🤖 ML Analysis", "🔤 NLP Analysis",
        "✅ Validation", "📥 Downloads"
    ])

    with tab1:
        _render_overview_tab(result, composite, tier, company)

    with tab2:
        _render_ml_tab(result)

    with tab3:
        _render_nlp_tab(result)

    with tab4:
        _render_validation_tab(result)

    with tab5:
        _render_downloads_tab(audit_id, company)


def _render_overview_tab(result, composite, tier, company):
    from ui.components.risk_gauge import render_risk_gauge, render_esg_score_metrics
    from ui.components.esg_heatmap import render_esg_heatmap

    render_risk_gauge(composite, tier)

    dim = result.get("dimensional_scores") or {}
    render_esg_score_metrics(
        composite=composite,
        risk_tier=tier,
        env_score=dim.get("environmental", {}).get("score") if isinstance(dim.get("environmental"), dict) else None,
        social_score=dim.get("social", {}).get("score") if isinstance(dim.get("social"), dict) else None,
        gov_score=dim.get("governance", {}).get("score") if isinstance(dim.get("governance"), dict) else None,
    )

    if dim:
        render_esg_heatmap(dim)

    st.divider()
    if result.get("executive_summary"):
        st.subheader("Executive Summary")
        st.info(result["executive_summary"])

    if result.get("investment_recommendation"):
        st.subheader("Investment Recommendation")
        st.success(result["investment_recommendation"])

    # Key metadata
    st.subheader("Audit Metadata")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Audit ID", result.get("audit_id", "")[:12] + "...")
        st.metric("Duration", f"{result.get('duration_seconds', 0):.1f}s")
    with col2:
        st.metric("ML Confidence", f"{(result.get('ml_prediction_confidence') or 0):.1%}")
        st.metric("Texts Analyzed", result.get("texts_analyzed", 0))
    with col3:
        ci = result.get("confidence_interval") or {}
        if isinstance(ci, dict):
            st.metric("CI Lower", ci.get("lower", "—"))
            st.metric("CI Upper", ci.get("upper", "—"))


def _render_ml_tab(result):
    from ui.components.esg_heatmap import render_feature_importance_chart

    st.subheader("ML Risk Model Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML Risk Score", f"{result.get('ml_risk_score', 0):.1f}/100")
    with col2:
        st.metric("ML Risk Tier", result.get("ml_risk_tier", "—"))
    with col3:
        st.metric("Confidence", f"{(result.get('ml_prediction_confidence') or 0):.1%}")

    fi = result.get("feature_importances") or {}
    if fi:
        render_feature_importance_chart(fi)

    # SHAP values
    shap = result.get("shap_contributions") or {}
    if shap:
        st.subheader("SHAP Feature Contributions")
        import pandas as pd
        rows = []
        for feat, data in shap.items():
            if isinstance(data, dict):
                rows.append({
                    "Feature": feat,
                    "Value": data.get("value", 0),
                    "SHAP Value": data.get("shap_value", 0),
                    "Importance": data.get("importance", 0),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # LIME
    lime = result.get("lime_explanation") or {}
    if lime:
        st.subheader("LIME Local Explanation")
        st.markdown(f"**Method:** `{lime.get('method', 'N/A')}`")
        st.markdown(f"**R²:** {lime.get('r_squared', 0):.4f}")
        st.info(lime.get("explanation_summary", ""))


def _render_nlp_tab(result):
    from ui.components.esg_heatmap import render_nlp_scores_radar, render_emerging_risks_table

    st.subheader("Zero-Shot NLP Analysis")

    nlp_agg = result.get("nlp_aggregate_scores") or {}
    if nlp_agg:
        render_nlp_scores_radar(nlp_agg)

    emerging = result.get("emerging_risks") or []
    if isinstance(emerging, list):
        emerging_dicts = []
        for r in emerging:
            if isinstance(r, dict):
                emerging_dicts.append(r)
            else:
                emerging_dicts.append(r.dict() if hasattr(r, "dict") else {})
    else:
        emerging_dicts = []

    st.subheader(f"Emerging Risks ({len(emerging_dicts)} detected)")
    render_emerging_risks_table(emerging_dicts)


def _render_validation_tab(result):
    from ui.components.esg_heatmap import render_validation_flags_summary

    st.subheader("Regulatory Validation")

    val_status = result.get("validation_status", "UNKNOWN")
    status_colors = {"PASS": "success", "WARNING": "warning", "FAIL": "error"}
    getattr(st, status_colors.get(val_status, "info"))(
        f"Validation Status: **{val_status}**"
    )

    reg_flags = result.get("regulatory_flags") or {}
    consistency = result.get("consistency_flags") or []

    # Convert to dicts
    reg_dicts = {}
    for k, flags in reg_flags.items():
        reg_dicts[k] = [f.dict() if hasattr(f, "dict") else f for f in flags]

    consistency_dicts = [f.dict() if hasattr(f, "dict") else f for f in consistency]

    render_validation_flags_summary(reg_dicts, consistency_dicts)

    # Risk signals
    signals = result.get("risk_signals") or []
    if signals:
        st.subheader("Risk Signals")
        import pandas as pd
        rows = [
            {
                "Type": s.type if hasattr(s, "type") else s.get("type"),
                "Severity": s.severity if hasattr(s, "severity") else s.get("severity"),
                "Message": s.message if hasattr(s, "message") else s.get("message"),
            }
            for s in signals
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_downloads_tab(audit_id: str, company: str) -> None:
    """Render report download buttons."""
    st.subheader("Download Reports")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Download JSON Report"):
            _download_report(audit_id, "json", company)

    with col2:
        if st.button("📝 Download Markdown Report"):
            _download_report(audit_id, "markdown", company)

    with col3:
        if st.button("📑 Download PDF Report"):
            _download_report(audit_id, "pdf", company)

    st.divider()
    st.caption(f"Audit ID: `{audit_id}`")


def _fetch_audit(audit_id: str) -> Optional[Dict[str, Any]]:
    """Fetch audit result from API."""
    try:
        resp = requests.get(f"{API_BASE}/audit/{audit_id}", timeout=15)
        if resp.ok:
            return resp.json()
        elif resp.status_code == 404:
            st.error(f"❌ Audit `{audit_id[:12]}...` not found.")
        else:
            st.error(f"API Error {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to ESG Auditor API. Is it running on port 8000?")
    except Exception as e:
        st.error(f"❌ Error: {e}")
    return None


def _download_report(audit_id: str, fmt: str, company: str) -> None:
    """Trigger report download."""
    url = f"{API_BASE}/report/{audit_id}/{fmt}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.ok:
            ext_map = {"json": "json", "markdown": "md", "pdf": "pdf"}
            filename = f"ESG_Audit_{company.replace(' ', '_')}_{audit_id[:8]}.{ext_map.get(fmt, fmt)}"
            mime_map = {
                "json": "application/json",
                "markdown": "text/markdown",
                "pdf": "application/pdf",
            }
            st.download_button(
                label=f"⬇️ Save {fmt.upper()}",
                data=resp.content,
                file_name=filename,
                mime=mime_map.get(fmt, "application/octet-stream"),
            )
        else:
            st.error(f"Report not available: {resp.text[:200]}")
    except Exception as e:
        st.error(f"Download failed: {e}")