"""Runtime configuration for the DreaMS inference web service."""

from dataclasses import dataclass
from pathlib import Path

# MSP is intentionally excluded: its loader is non-functional upstream in DreaMS.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mgf", ".mzml", ".mzxml", ".hdf5"})


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Immutable service settings with safe defaults.

    Values are data-driven so a deployment can override them without code changes
    (looser caps on terka, stricter caps once the service is public).
    """

    gpu_device: int = 1
    max_upload_bytes: int = 200 * 1024 * 1024
    max_spectra: int = 50_000
    job_retention_seconds: int = 24 * 3600
    job_store_max_bytes: int = 30 * 1024 * 1024 * 1024
    jobs_dir: Path = Path("/tmp/jozefov/dreams_inference/jobs")
    reference_library: Path = Path("data/MassSpecGym_DreaMS.hdf5")
    examples_dir: Path = Path("data/examples")


DEFAULT_SETTINGS = Settings()
