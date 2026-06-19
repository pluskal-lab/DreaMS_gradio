"""Slow end-to-end test for the annotation pipeline (needs the DreaMS model)."""

import shutil
from pathlib import Path

import pytest

from dreams_web.config import Settings
from dreams_web.service.annotate import annotate_upload
from dreams_web.service.reference_library import ReferenceLibrary
from dreams_web.service.search_backend import SearchBackend

_FIXTURE = Path(__file__).parent / "fixtures" / "example_5_spectra.mgf"


@pytest.mark.slow
def test_annotate_upload_end_to_end(tmp_path: Path) -> None:
    """Annotating the example against itself yields display rows and a TSV."""
    ref = tmp_path / "ref.mgf"
    upload = tmp_path / "upload.mgf"
    shutil.copy2(_FIXTURE, ref)
    shutil.copy2(_FIXTURE, upload)

    backend = SearchBackend(ReferenceLibrary(name="self", path=ref))
    result = annotate_upload(upload, backend, Settings(), work_dir=tmp_path)

    assert result.n_query_spectra >= 1
    assert result.n_matches >= 1
    assert result.display_rows
    assert b"\t" in result.tsv
