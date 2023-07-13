import abc
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, DatasetDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable, Optional, Dict
from discardbert.elimination_strategy import EliminationType, STR2ELIMINATION


class Training(abc.ABC):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 elimination_type: EliminationType, elimination_params: Dict, **kwargs):
        self.elimination = STR2ELIMINATION[elimination_type]()
        self.elimination_params = elimination_params
        self.model = model
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def train(self, optimizer: Optimizer, lr_scheduler: LRScheduler, dataset: Dataset,
              padding_fn: Optional[Callable], batch_size: int, num_epoch: int, logging_interval: int,
              use_wandb: bool, **kwargs):
        pass

    @abc.abstractmethod
    def eval(self, dataset: DatasetDict, compute_metrics: Callable, prefix: str, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, path: str):
        pass

    @abc.abstractmethod
    def path_information(self) -> str:
        pass