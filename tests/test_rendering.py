"""Tests for SMILES-to-SVG molecule rendering."""

from dreams_web.web.rendering import smiles_to_svg


def test_valid_smiles_renders_svg() -> None:
    """A valid SMILES renders inline SVG markup."""
    assert "<svg" in smiles_to_svg("CCO")


def test_invalid_or_empty_smiles_returns_empty() -> None:
    """Invalid or empty SMILES render to an empty string (no crash)."""
    assert smiles_to_svg("") == ""
    assert smiles_to_svg("definitely not a smiles ###") == ""
