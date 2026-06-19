"""Turn DreaMS search hits into a downloadable TSV and compact display rows."""

from dreams.definitions import SPECTRUM
from dreams.utils.spectra import unpad_peak_list
from pandas import DataFrame

# Columns surfaced in the results table; the full hit set stays in the TSV export.
_DISPLAY_COLUMNS = (
    "precursor_mz",
    "RT",
    "ref_smiles",
    "ref_name",
    "ref_adduct",
    "ref_precursor_mz",
    "DreaMS_similarity",
)


def to_tsv(hits: DataFrame) -> bytes:
    """
    Serialize search hits to TSV bytes, unpadding spectrum peak lists.
    Args:
        hits (DataFrame): Search results from the search backend.
    Returns:
        bytes: UTF-8 TSV with one row per (query, match).
    """
    out = hits.copy()
    # Spectrum columns hold zero-padded arrays; export them as compact peak lists.
    for col in (SPECTRUM, f"ref_{SPECTRUM}"):
        if col in out.columns:
            out[col] = out[col].apply(lambda peaks: unpad_peak_list(peaks).tolist())
    return out.to_csv(index=False, sep="\t").encode("utf-8")


def to_display_rows(hits: DataFrame) -> list[dict]:
    """
    Select a compact, human-readable subset of columns for the results table.
    Args:
        hits (DataFrame): Search results from the search backend.
    Returns:
        list[dict]: One dict per hit, limited to the present display columns.
    """
    present = [c for c in _DISPLAY_COLUMNS if c in hits.columns]
    return [dict(row) for row in hits[present].to_dict(orient="records")]
