"""
ESG Risk Gauge component for the Streamlit dashboard.
Renders an interactive gauge chart using Plotly.
"""

from typing import Optional
import streamlit as st


TIER_COLORS = {
    "LOW": "#27ae60",
    "MEDIUM": "#f39c12",
    "HIGH": "#e74c3c",
    "CRITICAL": "#8e44ad",
    "UNKNOWN": "#95a5a6",
}

TIER_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🔴",
    "CRITICAL": "🟣",
    "UNKNOWN": "⚪",
}


def render_risk_gauge(
    score: float,
    risk_tier: str,
    title: str = "Composite ESG Risk Score",
    height: int = 280,
) -> None:
    """
    Render an interactive gauge chart for the composite ESG risk score.

    Args:
        score: Composite ESG score (0–100). Higher = higher risk.
        risk_tier: Risk tier string (LOW/MEDIUM/HIGH/CRITICAL).
        title: Chart title.
        height: Chart height in pixels.
    """
    try:
        import plotly.graph_objects as go

        color = TIER_COLORS.get(risk_tier, "#95a5a6")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": f"{title}<br><span style='font-size:14px;color:{color}'>"
                            f"{TIER_EMOJI.get(risk_tier, '⚪')} {risk_tier} RISK</span>",
                    "font": {"size": 16},
                },
                delta={
                    "reference": 50,
                    "increasing": {"color": "#e74c3c"},
                    "decreasing": {"color": "#27ae60"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#333",
                        "tickvals": [0, 25, 50, 75, 100],
                        "ticktext": ["0", "25", "50", "75", "100"],
                    },
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#ccc",
                    "steps": [
                        {"range": [0, 25], "color": "#d5f5e3"},
                        {"range": [25, 50], "color": "#fef9e7"},
                        {"range": [50, 75], "color": "#fde8d8"},
                        {"range": [75, 100], "color": "#f9ebea"},
                    ],
                    "threshold": {
                        "line": {"color": "#2c3e50", "width": 3},
                        "thickness": 0.8,
                        "value": score,
                    },
                },
                number={"font": {"size": 36, "color": color}, "suffix": "/100"},
            )
        )

        fig.update_layout(
            height=height,
            margin=dict(l=20, r=20, t=60, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Segoe UI, Arial, sans-serif"},
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback: simple metric display
        _render_fallback_gauge(score, risk_tier, title)


def render_esg_score_metrics(
    composite: float,
    risk_tier: str,
    env_score: Optional[float] = None,
    social_score: Optional[float] = None,
    gov_score: Optional[float] = None,
) -> None:
    """
    Render a row of ESG score metric cards using st.metric.

    Args:
        composite: Composite ESG score (0–100).
        risk_tier: Risk tier string.
        env_score: Environmental dimensional score.
        social_score: Social dimensional score.
        gov_score: Governance dimensional score.
    """
    tier_emoji = TIER_EMOJI.get(risk_tier, "⚪")
    tier_color = TIER_COLORS.get(risk_tier, "#95a5a6")

    if all(s is not None for s in [env_score, social_score, gov_score]):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label=f"{tier_emoji} Composite Score",
                value=f"{composite:.1f}/100",
                delta=f"{risk_tier} Risk",
                delta_color="inverse" if risk_tier in {"HIGH", "CRITICAL"} else "normal",
            )
        with col2:
            st.metric(
                label="🌿 Environmental",
                value=f"{env_score:.1f}",
                delta="E Score",
                delta_color="off",
            )
        with col3:
            st.metric(
                label="👥 Social",
                value=f"{social_score:.1f}",
                delta="S Score",
                delta_color="off",
            )
        with col4:
            st.metric(
                label="🏛️ Governance",
                value=f"{gov_score:.1f}",
                delta="G Score",
                delta_color="off",
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{tier_emoji} Composite ESG Score",
                value=f"{composite:.1f}/100",
            )
        with col2:
            st.metric(label="Risk Tier", value=risk_tier)


def render_tier_legend() -> None:
    """Render a color-coded risk tier legend."""
    st.markdown("**Risk Tier Reference:**")
    cols = st.columns(4)
    tiers = [
        ("🟢 LOW", "Score 0–25", "#d5f5e3"),
        ("🟡 MEDIUM", "Score 25–50", "#fef9e7"),
        ("🔴 HIGH", "Score 50–75", "#fde8d8"),
        ("🟣 CRITICAL", "Score 75–100", "#f9ebea"),
    ]
    for col, (tier_label, score_range, bg) in zip(cols, tiers):
        with col:
            st.markdown(
                f"""
                <div style="background:{bg}; padding:8px 12px;
                            border-radius:6px; text-align:center;
                            font-size:12px;">
                  <strong>{tier_label}</strong><br>{score_range}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_fallback_gauge(score: float, tier: str, title: str) -> None:
    """Simple text-based fallback when Plotly is not available."""
    color = TIER_COLORS.get(tier, "#95a5a6")
    emoji = TIER_EMOJI.get(tier, "⚪")

    # ASCII-style progress bar
    filled = int(score / 5)
    bar = "█" * filled + "░" * (20 - filled)

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:8px; background:#f8f9fa;
                    border: 2px solid {color}; text-align:center;">
          <h3 style="color:{color}">{emoji} {title}</h3>
          <div style="font-size:36px; font-weight:bold; color:{color}">
            {score:.1f}<span style="font-size:16px;">/100</span>
          </div>
          <div style="font-family:monospace; letter-spacing:2px; color:{color}">
            [{bar}]
          </div>
          <div style="font-size:18px; font-weight:bold; color:{color}; margin-top:8px">
            {tier} RISK
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )