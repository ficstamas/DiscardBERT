from .base import Tokenizer
from typing import Dict, Literal, Type
from copy import deepcopy
import torch

CONLLType = Literal["ner", "pos"]


class CONLLTokenizer(Tokenizer):
    def __init__(self, tokenizer_name, keys, dataset, **kwargs):
        super().__init__(tokenizer_name, **kwargs)
        self.keys = keys
        self.dataset = dataset

    def tokenize(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], is_split_into_words=True, **self.kwargs
        )

        """
        wikiann: O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4), B-LOC (5), I-LOC (6).
        conll:   O (0), B-PER (1), I-PER (2), B-ORG (3)  I-ORG (4), B-LOC (5), I-LOC (6), B-MISC (7), I-MISC (8)
        model:   O (0), B-PER (3), I-PER (4), B-ORG (5), I-ORG (6), B-LOC (7), I-LOC (8), B-MISC (1), I-MISC (2).
        """

        labels = []
        task = self.keys

        id2label = {i: v for i,v in enumerate(self.dataset['train'].features[f'{task}_tags'].feature.names)}
        label2id = {v: k for k,v in id2label.items()}
        if task == "ner":
            if "B-PER" not in label2id:
                label2id["B-PER"] = label2id["I-PER"]

        data = [
            [label2id[id2label[x]] for x in y]
            for y in examples[f'{task}_tags']
        ]

        for i, label in enumerate(data):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either
                # the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if False else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


class NERTokenizer(CONLLTokenizer):
    def __init__(self, tokenizer_name, dataset, **kwargs):
        self.keys = "ner"
        super().__init__(tokenizer_name, self.keys, dataset, **kwargs)


class POSTokenizer(CONLLTokenizer):
    def __init__(self, tokenizer_name, dataset, **kwargs):
        self.keys = "pos"
        super().__init__(tokenizer_name, self.keys, dataset, **kwargs)


class ChunkTokenizer(CONLLTokenizer):
    def __init__(self, tokenizer_name, dataset, **kwargs):
        self.keys = "chunk"
        super().__init__(tokenizer_name, self.keys, dataset, **kwargs)


STR2CONLL_TASK: Dict[CONLLType, Type[CONLLTokenizer]] = {
    "ner": NERTokenizer,
    "pos": POSTokenizer,
    "chunk": ChunkTokenizer
}

