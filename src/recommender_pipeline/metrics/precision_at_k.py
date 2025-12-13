from ..interfaces.base_metric import BaseMetric
import numpy as np


class PrecisionAtK(BaseMetric):
    def __init__(self, k=10):
        self.k = k

    def name(self):
        return f"precision@{self.k}"

    def compute(self, model, interactions_df, k=None):
        if k is None:
            k = self.k

        num_users, _ = interactions_df.shape
        precisions = []

        for u in range(num_users):
            true_items = interactions_df[u].indices
            if len(true_items) == 0:
                continue

            pred_items = model.recommend(u, k=k)
            correct = len(set(pred_items) & set(true_items))
            precisions.append(correct / k)

        return float(np.mean(precisions))
