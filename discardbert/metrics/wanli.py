from evaluate import load
import numpy as np
from discardbert.tokenizer.wanli import WANLIType
from typing import get_args, Dict, Type
from .base import Metrics


class WANLIMetrics(Metrics):
    def __init__(self, **kwargs):
        self.metric = load("accuracy")

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)

        results = self.metric.compute(predictions=predictions, references=labels)
        return results


STR2WANLI_METRICS: Dict[WANLIType, Type[Metrics]] = {k: WANLIMetrics for k in get_args(WANLIType)}
