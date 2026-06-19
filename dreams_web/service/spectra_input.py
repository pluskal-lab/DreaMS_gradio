"""Loading, validation, and quality filtering of uploaded MS/MS spectra."""

from pathlib import Path

from dreams.definitions import CHARGE, PRECURSOR_MZ, SPECTRUM
from dreams.utils.data import MSData
from dreams.utils.dformats import assign_dformat
from dreams.utils.spectra import unpad_peak_list

from dreams_web.config import SUPPORTED_EXTENSIONS
from dreams_web.service.errors import EmptyResultError, InvalidInputError

# DreaMS quality band "A" = high-quality MS/MS (paper Fig. 2b).
_HIGH_QUALITY_FORMAT = "A"
# The released DreaMS model is trained on singly charged spectra only.
_MAX_ABS_CHARGE = 1


def validate_upload(path: Path, max_bytes: int) -> None:
    """
    Validate an uploaded MS/MS file's existence, type, and size.
    Args:
        path (Path): Path to the uploaded file.
        max_bytes (int): Maximum accepted file size in bytes.
    Raises:
        InvalidInputError: If the file is missing, oversized, or unsupported.
    """
    if not path.is_file():
        raise InvalidInputError(f"File not found: {path.name}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidInputError(f"Unsupported file type: {path.suffix or '(none)'}")
    size = path.stat().st_size
    if size > max_bytes:
        raise InvalidInputError(f"File too large: {size} bytes (max {max_bytes}).")


def load_spectra(path: Path) -> MSData:
    """
    Load an MS/MS file into DreaMS's in-memory representation.
    Args:
        path (Path): A supported MS/MS file (.mgf/.mzML/.mzXML/.hdf5).
    Returns:
        MSData: The loaded spectra (non-HDF5 inputs are auto-converted to HDF5).
    """
    try:
        # Append mode lets the search step store query embeddings into the object.
        return MSData.load(path, in_mem=True, mode="a")
    except Exception as exc:
        # Untrusted upload: surface any parse failure as a clean input error.
        raise InvalidInputError("Could not parse the file as MS/MS spectra.") from exc


def filter_high_quality(spectra: MSData, out_path: Path) -> MSData:
    """
    Keep only singly charged, high-quality spectra for reliable matching.
    Args:
        spectra (MSData): Loaded query spectra.
        out_path (Path): Destination HDF5 path for the filtered subset.
    Returns:
        MSData: The filtered subset.
    Raises:
        EmptyResultError: If no spectrum passes the filters.
    """
    columns = spectra.columns()
    keep: list[int] = []
    for i in range(len(spectra)):
        # Absent charge column => assume singly charged (DreaMS default), so keep it.
        charge = spectra.get_values(CHARGE, idx=i) if CHARGE in columns else 1
        # Skip multiply charged spectra: outside the model's training distribution.
        if abs(charge) > _MAX_ABS_CHARGE:
            continue
        peaks = unpad_peak_list(spectra.get_values(SPECTRUM, idx=i))
        # Skip spectra failing DreaMS quality band "A".
        if (
            assign_dformat(peaks, spectra.get_values(PRECURSOR_MZ, idx=i))
            != _HIGH_QUALITY_FORMAT
        ):
            continue
        keep.append(i)

    if not keep:
        raise EmptyResultError("No spectra passed the single-charge / quality filters.")
    # Append mode + in-memory so the search step can store embeddings into the subset.
    return spectra.form_subset(idx=keep, out_pth=out_path, mode="a", in_mem=True)
