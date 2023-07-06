from evaluate import load
import numpy as np
from discardbert.tokenizer.glue import GLUEType
from typing import get_args, Dict, Type
from .base import Metrics


class GLUEMetrics(Metrics):
    def __init__(self, **kwargs):
        self.dataset = kwargs.get("dataset")
        self.subset = kwargs.get("subset")
        self.metric = load(self.dataset, self.subset)

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)

        results = self.metric.compute(predictions=predictions, references=labels)
        return results


class STSBMetrics(GLUEMetrics):
    def compute_metrics(self, p):
        predictions, labels = p

        results = self.metric.compute(predictions=predictions, references=labels)
        return results


STR2GLUE_METRICS: Dict[GLUEType, Type[Metrics]] = {k: GLUEMetrics for k in get_args(GLUEType) if k != "stsb"}
STR2GLUE_METRICS["stsb"] = STSBMetrics
