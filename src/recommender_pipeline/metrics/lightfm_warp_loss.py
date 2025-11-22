from interfaces.base_metric import BaseMetric


class LightFMWARPLoss(BaseMetric):
    def name(self):
        return "lightfm_warp_loss"

    def compute(self, model, interactions, k=10):
        return float(model.model.get_training_loss())
