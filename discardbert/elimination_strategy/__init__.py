from .vanilla import ExactLayerEliminationStrategy, RangeBasedLayerEliminationStrategy, LayerEliminationStrategy
from typing import Literal, Dict, Type


EliminationType = Literal["exact", "range"]

STR2ELIMINATION: Dict[str, Type[LayerEliminationStrategy]] = {
    "exact": ExactLayerEliminationStrategy,
    "range": RangeBasedLayerEliminationStrategy
}
