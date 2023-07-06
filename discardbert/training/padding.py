import torch
from discardbert.model.task_type import ModelType
from typing import Dict, Callable
from torch import Tensor


def to_torch(batch, device):
    features = [x for x in batch.keys()]

    b = {}
    for f in features:
        b[f] = torch.tensor(batch[f], device=device)
    return b


def dynamic_padding_sequence_classification(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    features = ['input_ids', 'attention_mask', 'token_type_ids']
    max_len = max([len(b) for b in batch['input_ids']])

    b = {}
    for f in features:
        b[f] = torch.stack([torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=0) for t in batch[f]]).to(
            device)
    b['labels'] = batch['labels'].to(device)
    return b


def dynamic_padding_token_classification(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    features = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    max_len = max([len(b) for b in batch['input_ids']])

    b = {}
    for f in features:
        if f != 'labels':
            b[f] = torch.stack(
                [torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=0) for t in batch[f]]).to(device)
        else:
            b[f] = torch.stack(
                [torch.nn.functional.pad(t, (0, max_len - t.shape[-1]), value=-100) for t in batch[f]]).to(device)
    return b


STR2PADDING: Dict[ModelType, Callable[[Dict[str, Tensor], str], Dict[str, Tensor]]] = {
    "sequence": dynamic_padding_sequence_classification,
    "token": dynamic_padding_token_classification
}
