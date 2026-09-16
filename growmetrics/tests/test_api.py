"""
API endpoint integration tests using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestVideoRequest:
    def test_post_request_returns_202(self):
        with patch("app.generator.pipeline.run_pipeline", new_callable=AsyncMock):
            resp = client.post(
                "/api/v1/videos/request",
                json={"query": "How does the pH scale work?"},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "PENDING"

    def test_post_request_too_short_query(self):
        resp = client.post("/api/v1/videos/request", json={"query": "Hi"})
        assert resp.status_code == 422  # Pydantic min_length validation

    def test_get_job_not_found(self):
        resp = client.get("/api/v1/videos/nonexistent_id")
        assert resp.status_code == 404

    def test_list_jobs_returns_list(self):
        resp = client.get("/api/v1/videos")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_download_artifact_before_completion(self):
        """Downloading artifact of a non-completed job should return 409."""
        with patch("app.generator.pipeline.run_pipeline", new_callable=AsyncMock):
            post_resp = client.post(
                "/api/v1/videos/request",
                json={"query": "Why do atoms form covalent bonds?"},
            )
        job_id = post_resp.json()["job_id"]
        resp = client.get(f"/api/v1/videos/{job_id}/artifact")
        assert resp.status_code == 409
