"""Smoke tests for the dreams_web scaffold."""

import dreams_web
from dreams_web.config import DEFAULT_SETTINGS, SUPPORTED_EXTENSIONS


def test_version_is_set() -> None:
    """Verify the package exposes a non-empty version string."""
    assert dreams_web.__version__


def test_default_settings_are_safe() -> None:
    """Verify defaults target GPU 1 with positive, bounded caps."""
    assert DEFAULT_SETTINGS.gpu_device == 1
    assert DEFAULT_SETTINGS.max_upload_bytes > 0
    assert DEFAULT_SETTINGS.max_spectra > 0


def test_msp_is_not_accepted() -> None:
    """MSP must never be accepted: its loader is non-functional upstream."""
    assert ".msp" not in SUPPORTED_EXTENSIONS
    assert ".mgf" in SUPPORTED_EXTENSIONS
