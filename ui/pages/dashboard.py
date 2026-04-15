"""
Streamlit Dashboard Page – ESG Audit Overview.
Displays recent audits, score distributions, and system health.
"""

import streamlit as st
import requests
from typing import Any, Dict, List, Optional

from ui.components.risk_gauge import render_tier_legend, TIER_COLORS


API_BASE = "http://localhost:8000/v1"


def render_dashboard() -> None:
    """Render the main ESG audit dashboard page."""
    st.title("📊 ESG Auditor Dashboard")
    st.markdown("*Autonomous Multi-Agent ESG Risk Intelligence Platform*")

    # Health check
    with st.spinner("Checking system health..."):
        health = _fetch_health()

    _render_health_status(health)

    st.divider()

    render_tier_legend()
    st.divider()

    # Recent audits
    st.subheader("📋 Recent Audits")
    audits = _fetch_recent_audits(limit=20)

    if not audits:
        st.info("No audits found. Navigate to **Run Audit** to start your first ESG audit.")
        return

    _render_audit_summary_table(audits)
    st.divider()

    # Score distribution
    completed = [
        a for a in audits
        if a.get("status") == "completed" and a.get("composite_score") is not None
    ]

    if completed:
        _render_score_distribution(completed)


def _render_health_status(health: Optional[Dict[str, Any]]) -> None:
    """Render system health status row."""
    if not health:
        st.error("⚠️ Cannot connect to ESG Auditor API. Ensure the API is running.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        db_ok = health.get("database") == "ok"
        st.metric("🗄️ Database", "Online" if db_ok else "Error",
                  delta_color="normal" if db_ok else "inverse")
    with col2:
        redis_ok = "error" not in health.get("redis", "")
        st.metric("⚡ Redis/Celery", "Online" if redis_ok else "Offline",
                  delta_color="normal" if redis_ok else "off")
    with col3:
        model_ok = health.get("model_loaded", False)
        st.metric("🤖 ML Model", "Loaded" if model_ok else "Not Loaded",
                  delta_color="normal" if model_ok else "off")
    with col4:
        uptime = health.get("uptime_seconds", 0)
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m" if uptime > 60 else f"{uptime:.0f}s"
        st.metric("⏱️ API Uptime", uptime_str)


def _render_audit_summary_table(audits: List[Dict[str, Any]]) -> None:
    """Render a styled dataframe of recent audits."""
    import pandas as pd

    rows = []
    for a in audits:
        tier = a.get("risk_tier", "—")
        emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🟣"}.get(tier, "⚪")
        score = a.get("composite_score")
        rows.append({
            "Audit ID": a.get("audit_id", "")[:12] + "...",
            "Company": a.get("company_name", "—"),
            "Status": a.get("status", "—").upper(),
            "Risk Tier": f"{emoji} {tier}" if tier != "—" else "—",
            "Score": f"{score:.1f}/100" if score is not None else "—",
            "Created": a.get("created_at", "—"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_score_distribution(completed: List[Dict[str, Any]]) -> None:
    """Render a histogram of composite ESG scores from completed audits."""
    st.subheader("📈 Score Distribution")

    try:
        import plotly.graph_objects as go

        scores = [a["composite_score"] for a in completed]
        tiers = [a.get("risk_tier", "UNKNOWN") for a in completed]
        companies = [a.get("company_name", "Unknown") for a in completed]

        # Scatter: company vs score
        fig = go.Figure()

        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            tier_data = [(c, s) for c, s, t in zip(companies, scores, tiers) if t == tier]
            if tier_data:
                fig.add_trace(
                    go.Scatter(
                        x=[d[0] for d in tier_data],
                        y=[d[1] for d in tier_data],
                        mode="markers",
                        name=tier,
                        marker=dict(
                            color=TIER_COLORS.get(tier, "#95a5a6"),
                            size=12,
                            symbol="circle",
                        ),
                        hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
                    )
                )

        # Add threshold lines
        for threshold, label, color in [
            (25, "LOW→MEDIUM", "#f39c12"),
            (50, "MEDIUM→HIGH", "#e74c3c"),
            (75, "HIGH→CRITICAL", "#8e44ad"),
        ]:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )

        fig.update_layout(
            title="Company ESG Score Distribution",
            xaxis_title="Company",
            yaxis_title="Composite ESG Score",
            yaxis=dict(range=[0, 105]),
            height=380,
            xaxis_tickangle=-45,
            margin=dict(l=30, r=120, t=60, b=120),
            legend=dict(title="Risk Tier"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,249,250,1)",
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        import pandas as pd
        df = pd.DataFrame({"Company": [a["company_name"] for a in completed],
                           "Score": [a["composite_score"] for a in completed]})
        st.bar_chart(df.set_index("Company"))


def _fetch_health() -> Optional[Dict[str, Any]]:
    """Fetch health status from API."""
    try:
        resp = requests.get(f"http://localhost:8000/healthz", timeout=3)
        return resp.json() if resp.ok else None
    except Exception:
        return None


def _fetch_recent_audits(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch recent audits from API."""
    try:
        resp = requests.get(f"{API_BASE}/audit", params={"limit": limit}, timeout=5)
        if resp.ok:
            return resp.json().get("items", [])
    except Exception:
        pass
    return []