"""k-NN hyperparameter search: distance metrics and grid search."""
from .distance_metrics import METRIC_NAMES, get_metric_config

__all__ = ["METRIC_NAMES", "get_metric_config"]
