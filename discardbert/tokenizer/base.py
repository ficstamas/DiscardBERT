import abc
from transformers import AutoTokenizer


class Tokenizer(abc.ABC):
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.kwargs = kwargs

    @abc.abstractmethod
    def tokenize(self, examples):
        pass
