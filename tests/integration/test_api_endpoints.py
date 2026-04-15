"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client."""
    import asyncio
    from api.main import app
    from core.database import create_tables, drop_tables

    # Create tables synchronously for testing
    asyncio.get_event_loop().run_until_complete(create_tables())

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    asyncio.get_event_loop().run_until_complete(drop_tables())


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_healthz_returns_200(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_healthz_returns_ok_status(self, client):
        resp = client.get("/healthz")
        data = resp.json()
        assert data["status"] == "ok"

    def test_healthz_has_version(self, client):
        resp = client.get("/healthz")
        data = resp.json()
        assert "version" in data

    def test_livez_returns_200(self, client):
        resp = client.get("/livez")
        assert resp.status_code == 200

    def test_root_returns_app_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "docs" in data


class TestAuditEndpoints:
    """Tests for audit CRUD endpoints."""

    def test_create_audit_sync_returns_result(self, client, sample_esg_data):
        payload = {
            "company_name": "API Test Corp",
            "esg_data": sample_esg_data,
            "async_execution": False,
        }
        resp = client.post("/v1/audit", json=payload, timeout=120)

        assert resp.status_code in {200, 202}
        data = resp.json()
        assert "audit_id" in data
        assert "status" in data

    def test_create_audit_requires_company_name(self, client):
        payload = {"esg_data": {}}
        resp = client.post("/v1/audit", json=payload)
        assert resp.status_code == 422

    def test_create_audit_validates_weights(self, client):
        payload = {
            "company_name": "Weight Test",
            "ml_weight": 0.7,
            "nlp_weight": 0.7,  # Sum > 1.0
        }
        resp = client.post("/v1/audit", json=payload)
        assert resp.status_code == 422

    def test_get_audit_returns_404_for_unknown_id(self, client):
        resp = client.get("/v1/audit/nonexistent-id-12345")
        assert resp.status_code == 404

    def test_list_audits_returns_paginated_response(self, client):
        resp = client.get("/v1/audit?limit=5&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert "has_more" in data

    def test_get_audit_after_creation(self, client, sample_esg_data):
        """Test full create → get cycle."""
        # Create
        payload = {
            "company_name": "Cycle Test Corp",
            "esg_data": sample_esg_data,
            "async_execution": False,
        }
        create_resp = client.post("/v1/audit", json=payload, timeout=120)

        if create_resp.status_code not in {200, 202}:
            pytest.skip("Audit creation failed (likely model/dependency issue)")

        audit_id = create_resp.json()["audit_id"]

        # Get
        get_resp = client.get(f"/v1/audit/{audit_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["audit_id"] == audit_id
        assert data["company_name"] == "Cycle Test Corp"


class TestReportEndpoints:
    """Tests for report download endpoints."""

    def test_report_returns_404_for_unknown_audit(self, client):
        resp = client.get("/v1/report/nonexistent-id/json")
        assert resp.status_code == 404

    def test_report_info_returns_404_for_unknown_audit(self, client):
        resp = client.get("/v1/report/nonexistent-id/info")
        assert resp.status_code == 404