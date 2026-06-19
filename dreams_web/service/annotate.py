"""Annotate an upload: validate -> filter -> library search -> results."""

from dataclasses import dataclass
from pathlib import Path

from dreams_web.config import Settings
from dreams_web.service.errors import InvalidInputError
from dreams_web.service.result_view import to_display_rows, to_tsv
from dreams_web.service.search_backend import SearchBackend
from dreams_web.service.spectra_input import (
    filter_high_quality,
    load_spectra,
    validate_upload,
)


@dataclass(frozen=True, slots=True)
class AnnotationResult:
    """
    Outcome of annotating one uploaded file against a reference library.

    Args:
        display_rows (list[dict]): Curated per-hit rows for the results table.
        tsv (bytes): Full hits as a downloadable TSV (empty when there are no matches).
        n_query_spectra (int): Spectra remaining after quality filtering.
        n_matches (int): Number of library matches returned.
    """

    display_rows: list[dict]
    tsv: bytes
    n_query_spectra: int
    n_matches: int


def annotate_upload(
    upload_path: Path,
    backend: SearchBackend,
    settings: Settings,
    work_dir: Path,
    k: int = 1,
    threshold: float = 0.0,
) -> AnnotationResult:
    """
    Validate, filter, and annotate an uploaded MS/MS file against the library.
    Args:
        upload_path (Path): The uploaded MS/MS file.
        backend (SearchBackend): Pre-loaded search backend (model + library).
        settings (Settings): Service settings (size and spectra caps).
        work_dir (Path): Writable scratch directory for this request.
        k (int): Matches to return per query spectrum.
        threshold (float): Minimum DreaMS similarity for a match.
    Returns:
        AnnotationResult: Display rows, TSV, and counts.
    Raises:
        InvalidInputError: If the upload is missing, oversized, unsupported, or has
            more spectra than the configured cap.
        EmptyResultError: If no spectra survive quality filtering.
    """
    validate_upload(upload_path, max_bytes=settings.max_upload_bytes)

    spectra = load_spectra(upload_path)
    if len(spectra) > settings.max_spectra:
        raise InvalidInputError(
            f"Too many spectra: {len(spectra)} (max {settings.max_spectra})."
        )

    filtered = filter_high_quality(spectra, out_path=work_dir / "filtered.hdf5")
    hits = backend.query(filtered, k=k, threshold=threshold)
    if hits is None:
        # No reference spectrum cleared the similarity threshold.
        return AnnotationResult(
            display_rows=[], tsv=b"", n_query_spectra=len(filtered), n_matches=0
        )

    return AnnotationResult(
        display_rows=to_display_rows(hits),
        tsv=to_tsv(hits),
        n_query_spectra=len(filtered),
        n_matches=len(hits),
    )
