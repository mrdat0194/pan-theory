"""
5× repeated-run consistency benchmark for the 3 required chemistry concepts.

Runs each query 5 times against the live service and asserts:
  - 15/15 runs complete without FAILED status
  - All generated MP4 artifacts are non-empty
  - No silent failures (stale PENDING state)

Usage:
    pytest tests/test_consistency.py -v -s --timeout=600
    (requires the FastAPI server to be running on localhost:8000)
"""

import time
import pytest
import httpx

BASE_URL = "http://localhost:8000"
REQUIRED_QUERIES = [
    "How does the pH scale work?",
    "Why do atoms form covalent bonds?",
    "What is the difference between ionic and covalent bonding?",
]
RUNS_PER_QUERY = 5
POLL_INTERVAL_SECONDS = 5
MAX_WAIT_SECONDS = 300  # 5 minutes per job


def wait_for_completion(client: httpx.Client, job_id: str) -> dict:
    """Poll job status until COMPLETED or FAILED, or timeout."""
    deadline = time.time() + MAX_WAIT_SECONDS
    while time.time() < deadline:
        resp = client.get(f"{BASE_URL}/api/v1/videos/{job_id}", timeout=10)
        data = resp.json()
        if data["status"] in ("COMPLETED", "FAILED"):
            return data
        time.sleep(POLL_INTERVAL_SECONDS)
    return {"status": "TIMEOUT", "job_id": job_id}


@pytest.mark.parametrize("query", REQUIRED_QUERIES)
@pytest.mark.parametrize("run_index", range(1, RUNS_PER_QUERY + 1))
def test_consistency_run(query: str, run_index: int):
    """Submit a query and assert it completes successfully within the time limit."""
    with httpx.Client() as client:
        # Submit job
        post_resp = client.post(
            f"{BASE_URL}/api/v1/videos/request",
            json={"query": query},
            timeout=10,
        )
        assert post_resp.status_code == 202, f"Unexpected status {post_resp.status_code}"
        job_id = post_resp.json()["job_id"]

        print(f"\n[Run {run_index}/5] query={query!r}  job_id={job_id}")

        # Wait for completion
        result = wait_for_completion(client, job_id)
        status = result.get("status")

        print(
            f"  → status={status}  "
            f"fallback={result.get('used_fallback')}  "
            f"retries={result.get('retry_count')}  "
            f"cost≈${result.get('estimated_cost_usd')}"
        )

        assert status == "COMPLETED", (
            f"Run {run_index} for {query!r} ended with status={status}. "
            f"failure_reason={result.get('failure_reason')}  "
            f"detail={result.get('failure_detail')}"
        )

        # Verify artifact exists and is downloadable
        artifact_resp = client.get(
            f"{BASE_URL}/api/v1/videos/{job_id}/artifact",
            timeout=30,
        )
        assert artifact_resp.status_code == 200, "Artifact endpoint returned non-200"
        assert len(artifact_resp.content) > 1024, "Artifact is suspiciously small (<1KB)"
