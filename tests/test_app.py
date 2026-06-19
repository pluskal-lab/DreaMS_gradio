"""Slow test for the FastAPI app (needs the DreaMS model)."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dreams_web.config import Settings
from dreams_web.service.reference_library import ReferenceLibrary
from dreams_web.service.search_backend import SearchBackend
from dreams_web.web.app import create_app

_FIXTURE = Path(__file__).parent / "fixtures" / "example_5_spectra.mgf"


@pytest.mark.slow
def test_annotate_endpoint_renders_results(tmp_path: Path) -> None:
    """POST /annotate returns a results table for the example file."""
    ref = tmp_path / "ref.mgf"
    shutil.copy2(_FIXTURE, ref)

    app = create_app(Settings())
    # Inject a tiny backend so startup skips the heavy MassSpecGym load.
    app.state.backend = SearchBackend(ReferenceLibrary(name="self", path=ref))

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        with _FIXTURE.open("rb") as handle:
            response = client.post(
                "/annotate", files={"file": ("example.mgf", handle, "text/plain")}
            )

    assert response.status_code == 200
    assert "Matches" in response.text
