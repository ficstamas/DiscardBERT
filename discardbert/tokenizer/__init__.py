from .glue import STR2GLUE_TASK, GLUEType, Tokenizer
from typing import Literal, Dict, Type
from .wanli import STR2WANLI_TASK, WANLIType
from .conll import STR2CONLL_TASK, CONLLType


DatasetType = Literal["glue", "wanli", "conll"]
SubsetType = GLUEType | WANLIType | CONLLType


STR2TOKENIZER: Dict[DatasetType, Dict[SubsetType, Type[Tokenizer]]] = {
    "glue": STR2GLUE_TASK,
    "wanli": STR2WANLI_TASK,
    "conll": STR2CONLL_TASK
}
