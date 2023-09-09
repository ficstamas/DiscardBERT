from evaluate import load
import numpy as np
from discardbert.tokenizer.conll import CONLLType
from typing import get_args, Dict, Type
from .base import Metrics


class CONLLMetrics(Metrics):
    def __init__(self, **kwargs):
        self.id2label = kwargs.get("id2label")
        self.metric = load("seqeval")

    def compute_metrics(self, p):
        id2label = self.id2label
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        return results


STR2CONLL_METRICS: Dict[CONLLType, Type[Metrics]] = {k: CONLLMetrics for k in get_args(CONLLType)}

