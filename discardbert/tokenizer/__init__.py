from .glue import STR2GLUE_TASK, GLUEType, Tokenizer
from typing import Literal, Dict, Type
from .wanli import STR2WANLI_TASK, WANLIType


DatasetType = Literal["glue", "wanli"]
SubsetType = GLUEType | WANLIType


STR2TOKENIZER: Dict[DatasetType, Dict[SubsetType, Type[Tokenizer]]] = {
    "glue": STR2GLUE_TASK,
    "wanli": STR2WANLI_TASK
}
