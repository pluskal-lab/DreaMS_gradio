"""Slow tests for the FastAPI app (need the DreaMS model)."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dreams_web.config import Settings
from dreams_web.service.reference_library import ReferenceLibrary
from dreams_web.service.search_backend import SearchBackend
from dreams_web.web.app import create_app

_FIXTURE = Path(__file__).parent / "fixtures" / "example_5_spectra.mgf"


def _tiny_backend_app(tmp_path: Path, **settings_kwargs: object) -> object:
    """Build the app with a 5-spectrum backend so startup skips the heavy load."""
    ref = tmp_path / "ref.mgf"
    shutil.copy2(_FIXTURE, ref)
    app = create_app(Settings(**settings_kwargs))  # type: ignore[arg-type]
    app.state.backend = SearchBackend(ReferenceLibrary(name="self", path=ref))
    return app


@pytest.mark.slow
def test_annotate_endpoint_renders_results(tmp_path: Path) -> None:
    """POST /annotate returns a results table for the example file."""
    app = _tiny_backend_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        with _FIXTURE.open("rb") as handle:
            response = client.post(
                "/annotate", files={"file": ("example.mgf", handle, "text/plain")}
            )
    assert response.status_code == 200
    assert "Matches" in response.text


@pytest.mark.slow
def test_examples_endpoint_runs_prepared_file(tmp_path: Path) -> None:
    """POST /examples/{id} annotates a prepared example (no network download)."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    # Stand in for the 'drugs' example with the small fixture to avoid a download.
    shutil.copy2(_FIXTURE, examples_dir / "example_5_drugs_zhao2025.mgf")
    app = _tiny_backend_app(tmp_path, examples_dir=examples_dir)

    with TestClient(app) as client:
        response = client.post("/examples/drugs")
    assert response.status_code == 200
    assert "Matches" in response.text
