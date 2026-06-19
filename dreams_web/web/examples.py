"""Prepared example spectra offered as one-click trials."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

_HF_BASE = (
    "https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/auxiliary"
)


@dataclass(frozen=True, slots=True)
class Example:
    """
    A prepared example MS/MS file shown as a one-click trial.

    Args:
        example_id (str): URL-safe identifier used in the endpoint path.
        label (str): Button text shown in the UI.
        filename (str): File name under the examples dir (and at the HF base URL).
    """

    example_id: str
    label: str
    filename: str

    @property
    def url(self) -> str:
        """Hugging Face download URL for this example."""
        return f"{_HF_BASE}/{self.filename}"


EXAMPLES: tuple[Example, ...] = (
    Example("drugs", "5 drugs (quick)", "example_5_drugs_zhao2025.mgf"),
    Example("piper", "Piper extract — 2k spectra", "example_piper_2k_spectra.mgf"),
    Example(
        "piper_mzml", "Piper raw mzML (slow)", "Piper55-Leaf-r2_1uL_damiani2023.mzML"
    ),
)

_BY_ID = {example.example_id: example for example in EXAMPLES}


def get_example(example_id: str) -> Example | None:
    """Return the example with this id, or None if it is unknown."""
    return _BY_ID.get(example_id)


def ensure_example(example: Example, examples_dir: Path) -> Path:
    """
    Return the local path of an example, downloading it from HF if missing.
    Args:
        example (Example): The example descriptor.
        examples_dir (Path): Directory holding example files.
    Returns:
        Path: The local file path (downloaded on first use).
    """
    examples_dir.mkdir(parents=True, exist_ok=True)
    path = examples_dir / example.filename
    if not path.is_file():
        urlretrieve(example.url, path)
    return path
