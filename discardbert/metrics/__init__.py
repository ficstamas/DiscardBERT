from .glue import STR2GLUE_METRICS, Metrics
from .wanli import STR2WANLI_METRICS
from discardbert.tokenizer import DatasetType, SubsetType
from typing import Dict, Type


STR2METRICS: Dict[DatasetType, Dict[SubsetType, Type[Metrics]]] = {
    "glue": STR2GLUE_METRICS,
    "wanli": STR2WANLI_METRICS
}
