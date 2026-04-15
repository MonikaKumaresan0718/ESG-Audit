"""
ESG Heatmap and chart components for the Streamlit dashboard.
Visualizes feature importances, NLP scores, and regulatory flags.
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_feature_importance_chart(
    feature_importances: Dict[str, float],
    title: str = "ML Feature Importances (Top Risk Drivers)",
    max_features: int = 10,
    color: str = "#3498db",
) -> None:
    """
    Render a horizontal bar chart of ML feature importances.

    Args:
        feature_importances: Dict mapping feature names to importance scores.
        title: Chart title.
        max_features: Maximum number of features to display.
        color: Bar color hex string.
    """
    if not feature_importances:
        st.info("No feature importance data available.")
        return

    try:
        import plotly.graph_objects as go

        # Sort and truncate
        sorted_fi = sorted(
            feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:max_features]

        features = [f[0].replace("_", " ").title() for f in sorted_fi]
        values = [f[1] for f in sorted_fi]

        # Color bars by importance tier
        bar_colors = []
        for v in values:
            if v > 0.20:
                bar_colors.append("#e74c3c")
            elif v > 0.12:
                bar_colors.append("#f39c12")
            else:
                bar_colors.append("#3498db")

        fig = go.Figure(
            go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title={"text": title, "font": {"size": 15}},
            xaxis_title="Importance Score",
            yaxis={"categoryorder": "total ascending"},
            height=max(280, max_features * 32),
            margin=dict(l=10, r=60, t=50, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,249,250,1)",
        )

        fig.update_xaxes(showgrid=True, gridcolor="#eee")

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        _render_text_bar_chart(feature_importances, title, max_features)


def render_nlp_scores_radar(
    nlp_scores: Dict[str, float],
    title: str = "Zero-Shot NLP Risk Scores",
) -> None:
    """
    Render a radar chart of zero-shot NLP risk dimension scores.

    Args:
        nlp_scores: Dict of dimension → score (0–1 scale).
        title: Chart title.
    """
    if not nlp_scores:
        st.info("No NLP score data available.")
        return

    try:
        import plotly.graph_objects as go

        # Select key dimensions for radar
        display_dims = [
            "environmental", "social", "governance",
            "carbon emissions", "supply chain risk",
            "governance risk", "greenwashing",
        ]
        dims = []
        vals = []
        for dim in display_dims:
            if dim in nlp_scores:
                dims.append(dim.replace("_", " ").title())
                vals.append(float(nlp_scores[dim]))

        if not dims:
            # Use whatever we have
            dims = [k.replace("_", " ").title() for k in list(nlp_scores.keys())[:7]]
            vals = [float(v) for v in list(nlp_scores.values())[:7]]

        # Close the radar polygon
        dims_closed = dims + [dims[0]]
        vals_closed = vals + [vals[0]]

        fig = go.Figure(
            go.Scatterpolar(
                r=vals_closed,
                theta=dims_closed,
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.25)",
                line=dict(color="#3498db", width=2),
                marker=dict(size=6),
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.25, 0.50, 0.75, 1.0],
                    ticktext=["0.25", "0.50", "0.75", "1.0"],
                    gridcolor="#ddd",
                ),
                angularaxis=dict(gridcolor="#ddd"),
                bgcolor="rgba(248,249,250,1)",
            ),
            title={"text": title, "font": {"size": 15}},
            height=380,
            margin=dict(l=60, r=60, t=60, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.write("**NLP Risk Scores:**")
        for dim, score in list(nlp_scores.items())[:7]:
            st.write(f"- {dim}: {score:.4f}")


def render_esg_heatmap(
    dimensional_scores: Dict[str, Any],
    title: str = "ESG Dimensional Score Breakdown",
) -> None:
    """
    Render a heatmap comparing ML and NLP components per ESG dimension.

    Args:
        dimensional_scores: Dict from fusion result with E/S/G breakdowns.
        title: Chart title.
    """
    if not dimensional_scores:
        st.info("No dimensional score data available.")
        return

    try:
        import plotly.graph_objects as go

        dimensions = []
        ml_vals = []
        nlp_vals = []
        composite_vals = []

        for dim in ["environmental", "social", "governance"]:
            if dim in dimensional_scores:
                data = dimensional_scores[dim]
                dimensions.append(dim.title())
                ml_vals.append(float(data.get("ml_component", 0)))
                nlp_vals.append(float(data.get("nlp_component", 0)))
                composite_vals.append(float(data.get("score", 0)))

        if not dimensions:
            st.info("Dimensional score data format not recognized.")
            return

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="ML Component",
                x=dimensions,
                y=ml_vals,
                marker_color="#3498db",
                hovertemplate="<b>%{x} – ML</b><br>Score: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name="NLP Component",
                x=dimensions,
                y=nlp_vals,
                marker_color="#e67e22",
                hovertemplate="<b>%{x} – NLP</b><br>Score: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Composite Score",
                x=dimensions,
                y=composite_vals,
                mode="lines+markers",
                line=dict(color="#2c3e50", width=3, dash="dash"),
                marker=dict(size=10, color="#2c3e50"),
                hovertemplate="<b>%{x} – Composite</b><br>Score: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title={"text": title, "font": {"size": 15}},
            barmode="group",
            xaxis_title="ESG Dimension",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 100]),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=30, r=20, t=70, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,249,250,1)",
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eee")

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.write("**Dimensional Scores:**")
        for dim, data in dimensional_scores.items():
            if isinstance(data, dict):
                st.write(f"- **{dim.title()}**: {data.get('score', 0):.2f}")


def render_emerging_risks_table(
    emerging_risks: List[Dict[str, Any]],
    max_display: int = 10,
) -> None:
    """
    Render a styled table of emerging ESG risks detected by NLP.

    Args:
        emerging_risks: List of risk dicts with 'risk' and 'confidence' keys.
        max_display: Maximum number of risks to show.
    """
    if not emerging_risks:
        st.success("✅ No significant emerging risks detected by NLP analysis.")
        return

    import pandas as pd

    rows = []
    for risk in emerging_risks[:max_display]:
        conf = float(risk.get("confidence", 0))
        severity = "🔴 High" if conf > 0.6 else ("🟡 Medium" if conf > 0.4 else "🟢 Low")
        rows.append({
            "Risk Category": risk.get("risk", "Unknown"),
            "Confidence": f"{conf:.1%}",
            "Severity": severity,
            "Text Excerpt": (risk.get("text_excerpt", "")[:100] + "...") if risk.get("text_excerpt") else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_validation_flags_summary(
    regulatory_checks: Dict[str, List[Any]],
    consistency_flags: List[Any],
) -> None:
    """
    Render a summary of regulatory validation flags grouped by framework.

    Args:
        regulatory_checks: Dict with 'gri_flags', 'sasb_flags', 'tcfd_flags'.
        consistency_flags: List of pipeline consistency flags.
    """
    import pandas as pd

    all_flags = []

    framework_map = {
        "gri_flags": "GRI",
        "sasb_flags": "SASB",
        "tcfd_flags": "TCFD",
    }

    for key, framework in framework_map.items():
        for flag in regulatory_checks.get(key, []):
            if isinstance(flag, dict):
                all_flags.append({
                    "Framework": framework,
                    "Severity": flag.get("severity", "").upper(),
                    "Metric": flag.get("metric", "—"),
                    "Value": flag.get("value", "—"),
                    "Threshold": flag.get("threshold", "—"),
                    "Message": flag.get("message", ""),
                })
            else:
                # Pydantic model
                all_flags.append({
                    "Framework": framework,
                    "Severity": getattr(flag, "severity", "").upper(),
                    "Metric": getattr(flag, "metric", "—") or "—",
                    "Value": getattr(flag, "value", "—"),
                    "Threshold": getattr(flag, "threshold", "—"),
                    "Message": getattr(flag, "message", ""),
                })

    for flag in consistency_flags or []:
        if isinstance(flag, dict):
            all_flags.append({
                "Framework": "Consistency",
                "Severity": flag.get("severity", "").upper(),
                "Metric": flag.get("type", "—"),
                "Value": "—",
                "Threshold": "—",
                "Message": flag.get("message", ""),
            })

    if not all_flags:
        st.success("✅ All regulatory checks passed – no flags raised.")
        return

    df = pd.DataFrame(all_flags)

    # Count by severity
    critical_count = len(df[df["Severity"] == "CRITICAL"])
    warning_count = len(df[df["Severity"] == "WARNING"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Flags", len(all_flags))
    with col2:
        st.metric("Critical Flags", critical_count, delta_color="inverse")
    with col3:
        st.metric("Warning Flags", warning_count, delta_color="off")

    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_text_bar_chart(
    feature_importances: Dict[str, float],
    title: str,
    max_features: int,
) -> None:
    """ASCII-style fallback for feature importance when Plotly unavailable."""
    st.write(f"**{title}**")
    sorted_fi = sorted(
        feature_importances.items(), key=lambda x: x[1], reverse=True
    )[:max_features]

    for feature, importance in sorted_fi:
        bar_len = int(importance * 40)
        bar = "█" * bar_len
        st.text(f"{feature:<30} {bar} {importance:.4f}")