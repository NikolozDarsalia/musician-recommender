from ..interfaces.base_metric import BaseMetric
from lightfm.evaluation import auc_score


class LightFMAUC(BaseMetric):
    def name(self):
        return "lightfm_auc"

    def compute(self, model, interactions, k=10):
        return float(auc_score(model.model, interactions).mean())
