from interfaces.base_metric import BaseMetric
import numpy as np


class MAPAtK(BaseMetric):
    def __init__(self, k=10):
        self.k = k

    def name(self):
        return f"map@{self.k}"

    def compute(self, model, interactions_df, k=None):
        if k is None:
            k = self.k

        maps = []
        for u in range(interactions_df.shape[0]):
            true_items = interactions_df[u].indices
            if len(true_items) == 0:
                continue

            pred_items = model.recommend(u, k=k)
            score = 0
            hits = 0

            for i, p in enumerate(pred_items, start=1):
                if p in true_items:
                    hits += 1
                    score += hits / i

            maps.append(score / min(len(true_items), k))

        return float(np.mean(maps))
