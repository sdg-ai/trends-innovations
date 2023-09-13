import torch
import torchmetrics as tm
from collections import defaultdict
from torchmetrics import MetricCollection
from torchmetrics.utilities.data import _flatten_dict


class AvgDictMeter:
    """
    accumulates a dictionary of values and compute the average
    """
    def __init__(self):
        self.reset()
        self.values = defaultdict(float)
        self.n = 0

    def reset(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}


class TransformerMetricCollection(MetricCollection):
    """
    A collection of metrics for multi-class problem that can be updated and computed jointly for training and validation
    and testing.
    """
    def __init__(self, n_classes, device="cuda"):
        self.device = device
        metrics_collection = {
            "accuracy": tm.classification.MulticlassAccuracy(num_classes=n_classes).to(device),
            "precision_marco": tm.classification.MulticlassPrecision(average='macro', num_classes=n_classes).to(device),
            "precision_micro": tm.classification.MulticlassPrecision(average='micro', num_classes=n_classes).to(device),
            "recall_macro": tm.classification.MulticlassRecall(average='macro', num_classes=n_classes).to(device),
            "recall_micro": tm.classification.MulticlassRecall(average='micro', num_classes=n_classes).to(device),
            "f1_macro": tm.classification.MulticlassF1Score(average='macro', num_classes=n_classes).to(device),
            "f1_micro": tm.classification.MulticlassF1Score(average='micro', num_classes=n_classes).to(device)
        }
        super().__init__(metrics=metrics_collection)

        self.preds = torch.Tensor().to(device)
        self.targets = torch.Tensor().to(device)

    def compute(self) -> dict:
        """Compute the result for each metric in the collection."""
        res = {k: m(self.preds, self.targets) for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the metric state."""
        self.preds = self.preds.to(self.device)
        self.targets = self.targets.to(self.device)
        self.preds = torch.cat([self.preds, preds])
        self.targets = torch.cat([self.targets, targets])

    def reset(self):
        """Reset the metric state."""
        self.preds = torch.Tensor()
        self.targets = torch.Tensor()
        for metric in self.values():
            metric.reset()
