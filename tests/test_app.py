"""Tests for the FastAPI app: fast hardening checks + slow job-flow checks."""

import re
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dreams_web.config import Settings
from dreams_web.service.reference_library import ReferenceLibrary
from dreams_web.service.search_backend import SearchBackend
from dreams_web.web.app import create_app

_FIXTURE = Path(__file__).parent / "fixtures" / "example_5_spectra.mgf"


def _app(tmp_path: Path, with_model: bool, **settings_kwargs: Any) -> Any:
    """Build the app with a real tiny backend, or a dummy for rejection tests."""
    settings_kwargs.setdefault("jobs_dir", tmp_path / "jobs")
    app = create_app(Settings(**settings_kwargs))
    if with_model:
        ref = tmp_path / "ref.mgf"
        shutil.copy2(_FIXTURE, ref)
        app.state.backend = SearchBackend(ReferenceLibrary(name="self", path=ref))
    else:
        # Rejection tests never reach the worker, so the backend is never used.
        app.state.backend = object()
    return app


def _poll(client: Any, job_id: str, tries: int = 250) -> Any:
    """Poll a job until its view shows results or an error."""
    response = None
    for _ in range(tries):
        response = client.get(f"/jobs/{job_id}")
        if "Matches" in response.text or "Error" in response.text:
            return response
        time.sleep(0.1)
    return response


def test_unsupported_extension_rejected(tmp_path: Path) -> None:
    """A .txt upload is rejected with 400 before any work (no model)."""
    with TestClient(_app(tmp_path, with_model=False)) as client:
        r = client.post("/jobs", files={"file": ("data.txt", b"junk", "text/plain")})
    assert r.status_code == 400
    assert "Unsupported" in r.text


def test_oversized_upload_rejected(tmp_path: Path) -> None:
    """An upload over the cap is rejected with 400 (no model)."""
    app = _app(tmp_path, with_model=False, max_upload_bytes=100)
    with TestClient(app) as client:
        r = client.post("/jobs", files={"file": ("big.mgf", b"x" * 500, "text/plain")})
    assert r.status_code == 400
    assert "too large" in r.text.lower()


@pytest.mark.slow
def test_jobs_flow_renders_results(tmp_path: Path) -> None:
    """POST /jobs enqueues a job; polling it yields the results table."""
    with TestClient(_app(tmp_path, with_model=True)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        with _FIXTURE.open("rb") as fh:
            sub = client.post(
                "/jobs", files={"file": ("example.mgf", fh, "text/plain")}
            )
        assert sub.status_code == 200
        match = re.search(r"/jobs/([0-9a-f]+)", sub.text)
        assert match is not None
        result = _poll(client, match.group(1))
    assert "Matches" in result.text


@pytest.mark.slow
def test_examples_flow_runs(tmp_path: Path) -> None:
    """POST /examples/{id} enqueues a prepared example; polling yields results."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    shutil.copy2(_FIXTURE, examples_dir / "example_5_drugs_zhao2025.mgf")
    app = _app(tmp_path, with_model=True, examples_dir=examples_dir)
    with TestClient(app) as client:
        sub = client.post("/examples/drugs")
        match = re.search(r"/jobs/([0-9a-f]+)", sub.text)
        assert match is not None
        result = _poll(client, match.group(1))
    assert "Matches" in result.text
