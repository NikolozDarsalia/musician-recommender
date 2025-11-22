from interfaces.base_preprocessor import BasePreprocessor


class TagCleaner(BasePreprocessor):
    """
    - Converts tagID → tagValue
    - Normalizes text
    - Removes rare tags below threshold freq
    - Optionally clusters tags into semantic groups
    """
