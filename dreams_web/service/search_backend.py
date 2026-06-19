"""DreaMS-backed spectral library search."""

from dreams.api import DreaMSSearch
from dreams.utils.data import MSData
from pandas import DataFrame

from dreams_web.service.reference_library import ReferenceLibrary


class SearchBackend:
    """
    DreaMS spectral-library search over a fixed reference library.

    The reference embeddings and the DreaMS model load once at construction, so each
    query only embeds the uploaded spectra and runs a cosine nearest-neighbour search.
    """

    def __init__(self, library: ReferenceLibrary) -> None:
        """
        Load the reference library and the DreaMS model.
        Args:
            library (ReferenceLibrary): Reference spectra to search against.
        """
        self._library = library
        self._searcher = DreaMSSearch(ref_spectra=str(library.path))

    def query(
        self, spectra: MSData, k: int = 1, threshold: float = 0.0
    ) -> DataFrame | None:
        """
        Match query spectra against the reference library.
        Args:
            spectra (MSData): Validated, filtered query spectra.
            k (int): Number of top matches per query spectrum.
            threshold (float): Minimum DreaMS similarity to report a match.
        Returns:
            DataFrame | None: One row per (query, match) with ref_* annotation
                columns and a DreaMS_similarity score, or None if nothing matched.
        """
        return self._searcher.query(
            query_spectra=spectra, k=k, dreams_sim_thld=threshold
        )
