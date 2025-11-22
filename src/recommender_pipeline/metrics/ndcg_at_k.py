from interfaces.base_metric import BaseMetric
import numpy as np


class NDCGAtK(BaseMetric):
    def __init__(self, k=10):
        self.k = k

    def name(self):
        return f"ndcg@{self.k}"

    def compute(self, model, interaction_df, k=None):
        if k is None:
            k = self.k

        ndcgs = []
        for u in range(interaction_df.shape[0]):
            true_items = set(interaction_df[u].indices)
            if len(true_items) == 0:
                continue

            # Get top-k predicted items
            pred_items = model.recommend(u, k=k)

            # Compute DCG@k
            dcg = 0.0
            for rank, item in enumerate(pred_items, start=1):
                if item in true_items:
                    dcg += 1.0 / np.log2(rank + 1)

            # Compute IDCG@k (ideal DCG)
            # The best case is having all relevant items ranked at the top
            ideal_hits = min(len(true_items), k)
            idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        return float(np.mean(ndcgs))
