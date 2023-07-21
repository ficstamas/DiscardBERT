from .glue import GLUETokenizer
from typing import Dict, Literal, Type


WANLIType = Literal["wanli"]


class WANLITokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("premise", "hypothesis")
        super().__init__(tokenizer_name, self.keys, **kwargs)


STR2WANLI_TASK: Dict[WANLIType, Type[GLUETokenizer]] = {
    "wanli": WANLITokenizer,
}
