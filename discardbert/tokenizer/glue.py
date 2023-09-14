from .base import Tokenizer
from typing import Dict, Literal, Type


GLUEType = Literal["rte", "cola", "mrpc", "sst2", "qnli", "ax", "mnli", "qqp", "stsb", "wnli"]


class GLUETokenizer(Tokenizer):
    def __init__(self, tokenizer_name, keys, **kwargs):
        super().__init__(tokenizer_name, **kwargs)
        self.keys = keys

    def tokenize(self, examples):
        keys = self.keys

        sentence_1 = examples[keys[0]]
        sentence_2 = examples[keys[1]] if keys[1] is not None else None

        if 'dataset' in self.kwargs:
            del self.kwargs["dataset"]

        tokenized_inputs = self.tokenizer(
            sentence_1, sentence_2, **self.kwargs
        )
        tokenized_inputs['labels'] = examples['label']
        return tokenized_inputs


class RTETokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("sentence1", "sentence2")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class CoLATokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("sentence", None)
        super().__init__(tokenizer_name, self.keys, **kwargs)


class MRPCTokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("sentence1", "sentence2")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class SST2Tokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("sentence", None)
        super().__init__(tokenizer_name, self.keys, **kwargs)


class QNLITokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("question", "sentence")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class AXTokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("premise", "hypothesis")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class MNLITokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("premise", "hypothesis")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class QQPTokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("question1", "question2")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class STSBTokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("sentence1", "sentence2")
        super().__init__(tokenizer_name, self.keys, **kwargs)


class WNLITokenizer(GLUETokenizer):
    def __init__(self, tokenizer_name, **kwargs):
        self.keys = ("sentence1", "sentence2")
        super().__init__(tokenizer_name, self.keys, **kwargs)


STR2GLUE_TASK: Dict[GLUEType, Type[GLUETokenizer]] = {
    "rte": RTETokenizer,
    "cola": CoLATokenizer,
    "mrpc": MRPCTokenizer,
    "sst2": SST2Tokenizer,
    "qnli": QNLITokenizer,
    "ax": AXTokenizer,
    "mnli": MNLITokenizer,
    "qqp": QQPTokenizer,
    "stsb": STSBTokenizer,
    "wnli": WNLITokenizer,
}
