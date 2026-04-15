"""
Pytest configuration and shared fixtures for ESG Auditor tests.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Dict, Any

import pytest
import pytest_asyncio

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set testing environment BEFORE importing settings
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_esg.db")
os.environ.setdefault("VERBOSE_AGENTS", "False")
os.environ.setdefault("AUTO_TRAIN_MODEL", "False")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_esg_data() -> Dict[str, Any]:
    """Sample ESG metrics for testing."""
    return {
        "carbon_emissions": 245.7,
        "water_usage": 312.5,
        "board_diversity": 0.42,
        "employee_turnover": 0.18,
        "controversy_score": 4.2,
        "renewable_energy_pct": 0.35,
        "supply_chain_risk": 4.8,
        "esg_risk_label": 1,
    }


@pytest.fixture(scope="session")
def high_risk_esg_data() -> Dict[str, Any]:
    """High-risk ESG metrics for testing."""
    return {
        "carbon_emissions": 4500.0,
        "water_usage": 3800.0,
        "board_diversity": 0.15,
        "employee_turnover": 0.35,
        "controversy_score": 9.0,
        "renewable_energy_pct": 0.05,
        "supply_chain_risk": 9.5,
        "esg_risk_label": 2,
    }


@pytest.fixture(scope="session")
def low_risk_esg_data() -> Dict[str, Any]:
    """Low-risk ESG metrics for testing."""
    return {
        "carbon_emissions": 89.0,
        "water_usage": 180.0,
        "board_diversity": 0.58,
        "employee_turnover": 0.08,
        "controversy_score": 1.5,
        "renewable_energy_pct": 0.88,
        "supply_chain_risk": 2.0,
        "esg_risk_label": 0,
    }


@pytest.fixture(scope="session")
def sample_texts() -> list:
    """Sample ESG text snippets for NLP testing."""
    return [
        "The company has committed to carbon neutrality by 2030, reducing Scope 1 and 2 emissions by 50%.",
        "Employee safety incidents increased by 12% this year, raising concerns about workplace risk management.",
        "The board has approved a new diversity and inclusion policy with targets for gender parity by 2026.",
        "Water withdrawal from stressed regions increased significantly due to manufacturing expansion.",
        "The company faces regulatory scrutiny following a data privacy breach affecting 2 million users.",
    ]


@pytest.fixture(scope="session")
def mock_ml_result() -> Dict[str, Any]:
    """Mock ML model prediction result."""
    return {
        "risk_score_ml": 42.5,
        "risk_tier_ml": "MEDIUM",
        "predicted_class": 1,
        "class_probabilities": {"low": 0.25, "medium": 0.55, "high": 0.20},
        "feature_importances": {
            "carbon_emissions": 0.22,
            "controversy_score": 0.20,
            "water_usage": 0.15,
            "employee_turnover": 0.15,
            "board_diversity": 0.13,
            "renewable_energy_pct": 0.08,
            "supply_chain_risk": 0.07,
        },
        "features_used": [
            "carbon_emissions", "water_usage", "board_diversity",
            "employee_turnover", "controversy_score", "renewable_energy_pct",
            "supply_chain_risk",
        ],
        "model_version": "xgb_v1",
        "prediction_confidence": 0.75,
        "input_data_summary": {
            "carbon_emissions": 245.7,
            "water_usage": 312.5,
        },
    }


@pytest.fixture(scope="session")
def mock_zero_shot_result() -> Dict[str, Any]:
    """Mock zero-shot NLP analysis result."""
    return {
        "company_name": "Tesla Inc.",
        "texts_analyzed": 5,
        "aggregate_scores": {
            "environmental": 0.42,
            "social": 0.35,
            "governance": 0.28,
            "overall_nlp_risk": 0.35,
        },
        "label_scores": {
            "environmental risk": 0.42,
            "carbon emissions": 0.38,
            "social risk": 0.35,
            "governance risk": 0.28,
        },
        "emerging_risks": [
            {
                "risk": "transition risk",
                "confidence": 0.68,
                "source_text_index": 0,
                "text_excerpt": "carbon neutrality by 2030",
            }
        ],
        "text_level_results": [],
        "model_used": "facebook/bart-large-mnli",
        "analysis_complete": True,
    }


@pytest_asyncio.fixture
async def async_test_db():
    """Provide an async test database session."""
    from core.database import AsyncSessionLocal, create_tables, drop_tables

    await create_tables()
    async with AsyncSessionLocal() as session:
        yield session

    await drop_tables()
    # Cleanup test DB file
    if os.path.exists("test_esg.db"):
        os.remove("test_esg.db")


@pytest.fixture
def api_client():
    """Provide a FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        yield client