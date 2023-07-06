from .glue import STR2GLUE_TASK, GLUEType, Tokenizer
from typing import Literal, Dict, Type


DatasetType = Literal["glue"]
SubsetType = GLUEType


STR2TOKENIZER: Dict[DatasetType, Dict[SubsetType, Type[Tokenizer]]] = {
    "glue": STR2GLUE_TASK
}
