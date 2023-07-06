from .glue import STR2GLUE_METRICS, Metrics
from discardbert.tokenizer import DatasetType, SubsetType
from typing import Dict, Type


STR2METRICS: Dict[DatasetType, Dict[SubsetType, Type[Metrics]]] = {
    "glue": STR2GLUE_METRICS
}
