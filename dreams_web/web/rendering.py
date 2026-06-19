"""Render molecule SMILES to inline SVG for the results table."""

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

_IMG_PX = 130


@lru_cache(maxsize=4096)
def smiles_to_svg(smiles: str) -> str:
    """
    Render a SMILES string to inline SVG markup.
    Args:
        smiles (str): A molecule SMILES string.
    Returns:
        str: Inline SVG, or an empty string if the SMILES is missing or invalid.
    """
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(_IMG_PX, _IMG_PX)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
