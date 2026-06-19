"""Tests for the reference library descriptor."""

from pathlib import Path

from dreams_web.config import Settings
from dreams_web.service.reference_library import ReferenceLibrary, massspecgym_library


def test_exists_reflects_file_presence(tmp_path: Path) -> None:
    """exists() is True only when the library file is on disk."""
    missing = ReferenceLibrary(name="x", path=tmp_path / "nope.hdf5")
    assert not missing.exists()

    present_path = tmp_path / "lib.hdf5"
    present_path.write_bytes(b"")
    assert ReferenceLibrary(name="x", path=present_path).exists()


def test_massspecgym_library_uses_settings_path(tmp_path: Path) -> None:
    """The MassSpecGym descriptor takes its path from settings."""
    settings = Settings(reference_library=tmp_path / "ref.hdf5")
    library = massspecgym_library(settings)
    assert library.name == "MassSpecGym"
    assert library.path == tmp_path / "ref.hdf5"
