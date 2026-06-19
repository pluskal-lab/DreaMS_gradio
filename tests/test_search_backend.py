"""Slow integration test for the DreaMS search backend (needs the DreaMS model)."""

import shutil
from pathlib import Path

import pytest

from dreams_web.service.reference_library import ReferenceLibrary
from dreams_web.service.search_backend import SearchBackend
from dreams_web.service.spectra_input import load_spectra

_FIXTURE = Path(__file__).parent / "fixtures" / "example_5_spectra.mgf"


@pytest.mark.slow
def test_self_match_returns_perfect_hits(tmp_path: Path) -> None:
    """Self-searching the example spectra yields near-perfect self-matches."""
    ref = tmp_path / "ref.mgf"
    query = tmp_path / "query.mgf"
    shutil.copy2(_FIXTURE, ref)
    shutil.copy2(_FIXTURE, query)

    backend = SearchBackend(ReferenceLibrary(name="self", path=ref))
    hits = backend.query(load_spectra(query), k=1, threshold=0.0)

    assert hits is not None
    assert len(hits) == 5
    # Identical spectra embed identically => cosine self-similarity ~1.
    assert hits["DreaMS_similarity"].min() > 0.99
