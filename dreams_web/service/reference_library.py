"""Reference spectral libraries to match query spectra against."""

from dataclasses import dataclass
from pathlib import Path

from dreams_web.config import Settings


@dataclass(frozen=True, slots=True)
class ReferenceLibrary:
    """
    A named library of reference spectra with precomputed DreaMS embeddings.

    Args:
        name (str): Human-readable name, surfaced in results.
        path (Path): Path to the DreaMS HDF5 reference file.
    """

    name: str
    path: Path

    def exists(self) -> bool:
        """Return True if the library file is present on disk."""
        return self.path.is_file()


def massspecgym_library(settings: Settings) -> ReferenceLibrary:
    """
    Build the default MassSpecGym reference library from settings.
    Args:
        settings (Settings): Service settings holding the reference path.
    Returns:
        ReferenceLibrary: The MassSpecGym library descriptor.
    """
    return ReferenceLibrary(name="MassSpecGym", path=settings.reference_library)
