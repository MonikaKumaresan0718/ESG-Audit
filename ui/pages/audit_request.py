"""
Streamlit Audit Request Page – Submit a new ESG audit.
"""

import time
import streamlit as st
import requests
from typing import Any, Dict, Optional


API_BASE = "http://localhost:8000/v1"


def render_audit_request() -> None:
    """Render the ESG audit submission form."""
    st.title("🔍 Run ESG Audit")
    st.markdown("Submit a new ESG audit for any company. The system will run the full multi-agent pipeline.")

    with st.form("audit_form", clear_on_submit=False):
        st.subheader("Company Information")
        company_name = st.text_input(
            "Company Name *",
            placeholder="e.g. Tesla Inc.",
            help="Enter the full company name to audit",
        )

        st.subheader("ESG Metrics (Optional – uses sample data if blank)")
        col1, col2 = st.columns(2)

        with col1:
            carbon = st.number_input("Carbon Emissions (metric tons CO2e)", min_value=0.0, value=245.7, step=10.0)
            water = st.number_input("Water Usage (million liters)", min_value=0.0, value=312.5, step=10.0)
            diversity = st.slider("Board Diversity (ratio)", 0.0, 1.0, 0.42, 0.01)
            renewable = st.slider("Renewable Energy %", 0.0, 1.0, 0.35, 0.01)

        with col2:
            turnover = st.slider("Employee Turnover Rate", 0.0, 1.0, 0.18, 0.01)
            controversy = st.slider("Controversy Score (0–10)", 0.0, 10.0, 4.2, 0.1)
            supply_risk = st.slider("Supply Chain Risk (0–10)", 0.0, 10.0, 4.8, 0.1)

        st.subheader("Fusion Configuration")
        col3, col4 = st.columns(2)
        with col3:
            ml_weight = st.slider("ML Model Weight", 0.1, 0.9, 0.6, 0.05,
                                   help="Weight for ML risk score in hybrid fusion")
        with col4:
            nlp_weight = round(1.0 - ml_weight, 2)
            st.metric("NLP Model Weight", nlp_weight)

        st.subheader("Options")
        async_exec = st.checkbox("Async Execution (Celery)", value=True,
                                  help="Run via Celery queue; poll for results")
        fetch_news = st.checkbox("Fetch News Articles", value=False,
                                  help="Requires NEWS_API_KEY in .env")

        submitted = st.form_submit_button("🚀 Run ESG Audit", type="primary")

    if submitted:
        if not company_name.strip():
            st.error("❌ Please enter a company name.")
            return

        payload = {
            "company_name": company_name.strip(),
            "esg_data": {
                "carbon_emissions": carbon,
                "water_usage": water,
                "board_diversity": diversity,
                "employee_turnover": turnover,
                "controversy_score": controversy,
                "renewable_energy_pct": renewable,
                "supply_chain_risk": supply_risk,
            },
            "fetch_news": fetch_news,
            "async_execution": async_exec,
            "ml_weight": ml_weight,
            "nlp_weight": nlp_weight,
        }

        with st.spinner(f"Submitting ESG audit for **{company_name}**..."):
            result = _submit_audit(payload)

        if result:
            audit_id = result.get("audit_id")
            status = result.get("status")

            st.success(f"✅ Audit submitted! **Audit ID:** `{audit_id}`")
            st.session_state["last_audit_id"] = audit_id
            st.session_state["last_company"] = company_name

            if status == "completed" and result.get("result"):
                st.subheader("📊 Immediate Results")
                _render_inline_result(result["result"])
            else:
                st.info(
                    f"Audit is **{status.upper()}**. "
                    f"Navigate to **View Report** and enter Audit ID `{audit_id}` to check results."
                )

                # Auto-poll for sync result
                if not async_exec:
                    _poll_for_result(audit_id, company_name)


def _submit_audit(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Submit audit to API."""
    try:
        resp = requests.post(f"{API_BASE}/audit", json=payload, timeout=300)
        if resp.ok:
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text[:300]}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to ESG Auditor API. Is it running on port 8000?")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    return None


def _poll_for_result(audit_id: str, company_name: str, max_polls: int = 30) -> None:
    """Poll the API for audit completion."""
    progress = st.progress(0)
    status_text = st.empty()

    for i in range(max_polls):
        time.sleep(3)
        progress.progress((i + 1) / max_polls)
        status_text.text(f"Polling for results... ({i+1}/{max_polls})")

        try:
            resp = requests.get(f"{API_BASE}/audit/{audit_id}", timeout=10)
            if resp.ok:
                data = resp.json()
                poll_status = data.get("status")

                if poll_status == "completed":
                    progress.progress(1.0)
                    status_text.empty()
                    st.success("✅ Audit completed!")
                    _render_inline_result(data)
                    return
                elif poll_status == "failed":
                    progress.progress(1.0)
                    status_text.empty()
                    st.error(f"❌ Audit failed: {data.get('error_message', 'Unknown error')}")
                    return
        except Exception:
            pass

    progress.progress(1.0)
    status_text.empty()
    st.warning(f"⏳ Audit still running. Check **View Report** with ID: `{audit_id}`")


def _render_inline_result(result: Dict[str, Any]) -> None:
    """Render a quick summary of completed audit results."""
    from ui.components.risk_gauge import render_esg_score_metrics, render_risk_gauge

    composite = result.get("composite_esg_score") or 0
    tier = result.get("risk_tier") or "UNKNOWN"
    dim = result.get("dimensional_scores") or {}

    render_risk_gauge(composite, tier, height=250)
    render_esg_score_metrics(
        composite=composite,
        risk_tier=tier,
        env_score=dim.get("environmental", {}).get("score") if isinstance(dim.get("environmental"), dict) else None,
        social_score=dim.get("social", {}).get("score") if isinstance(dim.get("social"), dict) else None,
        gov_score=dim.get("governance", {}).get("score") if isinstance(dim.get("governance"), dict) else None,
    )

    if result.get("executive_summary"):
        st.info(f"**Summary:** {result['executive_summary'][:400]}...")

    if result.get("investment_recommendation"):
        st.markdown(f"**📋 Recommendation:** {result['investment_recommendation']}")