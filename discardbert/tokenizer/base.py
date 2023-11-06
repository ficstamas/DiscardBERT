import abc
from transformers import AutoTokenizer


class Tokenizer(abc.ABC):
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        kw = {}
        if "add_prefix_space" in kwargs:
            kw["add_prefix_space"] = kwargs["add_prefix_space"]
            kwargs.pop("add_prefix_space")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kw)
        self.kwargs = kwargs

    @abc.abstractmethod
    def tokenize(self, examples):
        pass
