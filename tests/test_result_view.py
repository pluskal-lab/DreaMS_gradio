"""Tests for result formatting (TSV export and display rows)."""

import numpy as np
from pandas import DataFrame

from dreams_web.service.result_view import to_display_rows, to_tsv


def _fake_hits() -> DataFrame:
    """Build a minimal hits DataFrame mimicking the search backend output."""
    # Shape (2, n_peaks): row 0 = m/z, row 1 = intensity, with a trailing zero pad.
    padded = np.array([[100.0, 200.0, 0.0], [1.0, 0.5, 0.0]])
    return DataFrame(
        {
            "precursor_mz": [201.1],
            "RT": [42.0],
            "ref_smiles": ["CCO"],
            "ref_name": ["ethanol"],
            "ref_adduct": ["[M+H]+"],
            "ref_precursor_mz": [201.0],
            "DreaMS_similarity": [0.97],
            "spectrum": [padded],
            "ref_spectrum": [padded],
        }
    )


def test_to_display_rows_selects_present_columns() -> None:
    """Display rows expose the curated columns and drop spectra arrays."""
    rows = to_display_rows(_fake_hits())
    assert len(rows) == 1
    assert rows[0]["ref_smiles"] == "CCO"
    assert "spectrum" not in rows[0]


def test_to_tsv_is_tab_separated() -> None:
    """TSV export is tab-separated and includes the reference annotation."""
    tsv = to_tsv(_fake_hits()).decode("utf-8")
    assert "\t" in tsv
    assert "ethanol" in tsv
