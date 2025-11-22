from interfaces.base_preprocessor import BasePreprocessor


class FeatureScaler(BasePreprocessor):
    """
    MinMax scaling (or log-scale + MinMax) for continuous features.
    """
