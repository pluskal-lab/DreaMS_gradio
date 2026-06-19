"""Tests for the spectra input layer (load / validate / filter)."""

import shutil
from pathlib import Path

import pytest

from dreams_web.service.errors import InvalidInputError
from dreams_web.service.spectra_input import (
    filter_high_quality,
    load_spectra,
    validate_upload,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "example_5_spectra.mgf"


def test_validate_accepts_supported_file() -> None:
    """A real .mgf within the size cap passes validation."""
    validate_upload(_FIXTURE, max_bytes=10_000_000)


def test_validate_rejects_unsupported_extension(tmp_path: Path) -> None:
    """An unsupported extension (e.g. .msp) is rejected."""
    bad = tmp_path / "spectra.msp"
    bad.write_text("")
    with pytest.raises(InvalidInputError):
        validate_upload(bad, max_bytes=10_000_000)


def test_validate_rejects_oversized_file() -> None:
    """A file larger than the cap is rejected."""
    with pytest.raises(InvalidInputError):
        validate_upload(_FIXTURE, max_bytes=10)


def test_load_and_filter_example(tmp_path: Path) -> None:
    """The 5-spectrum example loads and yields high-quality spectra."""
    # Copy to a temp dir so the auto-generated sibling .hdf5 lands there.
    local = tmp_path / _FIXTURE.name
    shutil.copy2(_FIXTURE, local)

    spectra = load_spectra(local)
    assert len(spectra) == 5

    filtered = filter_high_quality(spectra, out_path=tmp_path / "filtered.hdf5")
    assert len(filtered) >= 1
