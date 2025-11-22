from interfaces.base_preprocessor import BasePreprocessor


class FillMissingSpotify(BasePreprocessor):
    """
    Imputes missing Spotify popularity/audio features using:
    - LastFM listen counts
    - global means
    - per-genre averages
    """
