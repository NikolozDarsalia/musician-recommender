from sklearn.pipeline import Pipeline as skPipeline
from sklearn.base import BaseEstimator

class Pipeline(skPipeline, BaseEstimator):
    """
    BasePipeline extends sklearn's Pipeline for custom pipeline logic.
    Inherits all functionality from sklearn.pipeline.Pipeline.
    """
    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps=steps, memory=memory, verbose=verbose)
