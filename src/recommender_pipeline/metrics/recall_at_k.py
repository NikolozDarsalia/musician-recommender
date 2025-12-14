from ..interfaces.base_metric import BaseMetric
import numpy as np


class RecallAtK(BaseMetric):
    def __init__(self, k=10):
        self.k = k

    def name(self):
        return f"recall@{self.k}"

    def compute(self, model, interaction_df, k=None):
        if k is None:
            k = self.k

        recalls = []
        for u in range(interaction_df.shape[0]):
            true_items = interaction_df[u].indices
            if len(true_items) == 0:
                continue

            pred_items = model.recommend(u, k=k)
            correct = len(set(pred_items) & set(true_items))
            recalls.append(correct / len(true_items))

        return float(np.mean(recalls))
