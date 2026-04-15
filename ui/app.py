"""
Streamlit application entry point for the ESG Auditor dashboard.
Handles page routing and global configuration.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Auditor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/esg-auditor",
        "Report a bug": "https://github.com/your-org/esg-auditor/issues",
        "About": "**ESG Auditor** – Autonomous Multi-Agent ESG Risk Intelligence v1.0.0",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp { background-color: #f8f9fa; }
        .stMetric { background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .stButton > button { border-radius: 6px; font-weight: 600; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; }
        .stDataFrame { border-radius: 8px; }
        div[data-testid="stSidebar"] { background-color: #2c3e50; }
        div[data-testid="stSidebar"] .stMarkdown { color: white; }
        div[data-testid="stSidebar"] label { color: #ecf0f1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
def render_sidebar() -> str:
    """Render sidebar with navigation and branding."""
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 10px 0 20px;'>
                <div style='font-size: 42px;'>🌍</div>
                <div style='color: white; font-size: 20px; font-weight: 700;'>ESG Auditor</div>
                <div style='color: #bdc3c7; font-size: 11px;'>Multi-Agent Risk Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        page = st.radio(
            "Navigate",
            options=["📊 Dashboard", "🔍 Run Audit", "📄 View Report"],
            label_visibility="collapsed",
        )

        st.divider()

        # API status indicator
        try:
            import requests
            resp = requests.get("http://localhost:8000/healthz", timeout=2)
            if resp.ok:
                st.markdown("🟢 **API:** Online")
            else:
                st.markdown("🔴 **API:** Error")
        except Exception:
            st.markdown("🔴 **API:** Offline")

        st.divider()

        st.markdown(
            """
            <div style='color: #7f8c8d; font-size: 10px; text-align: center;'>
                ESG Auditor v1.0.0<br>
                GRI · SASB · TCFD<br>
                XGBoost + BART-large-MNLI
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ── Router ────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main application router."""
    page = render_sidebar()

    if page == "📊 Dashboard":
        from ui.pages.dashboard import render_dashboard
        render_dashboard()

    elif page == "🔍 Run Audit":
        from ui.pages.audit_request import render_audit_request
        render_audit_request()

    elif page == "📄 View Report":
        from ui.pages.report_viewer import render_report_viewer
        render_report_viewer()


if __name__ == "__main__":
    main()