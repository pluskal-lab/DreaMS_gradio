"""Domain exceptions for the DreaMS web service."""


class DreamsWebError(Exception):
    """Base class for all dreams_web service errors."""


class InvalidInputError(DreamsWebError):
    """Raised when an uploaded file is missing, too large, or an unsupported type."""


class EmptyResultError(DreamsWebError):
    """Raised when no spectra survive validation or quality filtering."""
