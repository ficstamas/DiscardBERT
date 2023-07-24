import copy
import os
import pickle

from transformers import TrainingArguments, Trainer, IntervalStrategy, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
import wandb
from typing import Callable, Optional
from .padding import to_torch
from .base import Training
from discardbert.utils.layers import assign_new_layers, retrieve_layers


class Simple(Training):
    def train(self, optimizer: Optimizer, lr_scheduler: LRScheduler, dataset: DatasetDict,
              padding_fn: Optional[Callable], batch_size: int, num_epoch: int, logging_interval: int,
              use_wandb: bool, **kwargs):
        """
        Trains the provided model with the set configuration
        :param optimizer: Optimizer
        :param lr_scheduler: Learning-rate Scheduler
        :param dataset: Dataset
        :param padding_fn: Padding function with two parameters: batch and device
        :param batch_size: Batch size
        :param num_epoch: Number of Epoch
        :param logging_interval: Logging Interval
        :param use_wandb: Whether to use Wandb
        :return:
        """
        elimination_applied = kwargs.get("elimination_applied", False)
        device = kwargs.get("device", "cpu")

        if not elimination_applied:
            self.apply_elimination()
        model = self.model.to(device)
        model.train()
        dataset = copy.deepcopy(dataset["train"])
        dataset.set_format("torch")

        steps = len(dataset) // batch_size
        progress = tqdm.tqdm(desc="Training Model: loss=None", total=steps * num_epoch, unit="steps")

        step_counter = 0
        mean_losses = []
        losses = []

        for e in range(num_epoch):
            dataloader = dataset.shuffle(e)
            for ix in range(0, len(dataloader), batch_size):
                data = dataloader[ix:ix + batch_size]
                batch = padding_fn(data, model.device) if padding_fn is not None else to_torch(data, model.device)

                student_output = model(**batch)

                student_output.loss.backward()
                losses.append(student_output.loss.detach().item())
                if step_counter % logging_interval == 0 and step_counter != 0:
                    mean_loss = np.mean(losses)
                    mean_losses.append(mean_loss.item())
                    if use_wandb:
                        wandb.log({"train/loss": mean_loss})
                    progress.set_description(f"Training Model: loss={mean_loss.item()}")
                    losses = []

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress.update()
                step_counter += 1
        progress.close()
        self.model = model.cpu()
        return mean_losses

    def eval(self, dataset: DatasetDict, compute_metrics: Callable, prefix: str = "eval", **kwargs):
        """
        Evaluates model on given dataset and evaluation function
        :param dataset: Dataset containing train, validation and test splits
        :param compute_metrics: Evaluation function with one parameter
        :param prefix: Prefix string for metrics
        :return:
        """
        model = self.model
        tokenizer = self.tokenizer

        model.eval()
        training_args = TrainingArguments("~/temp/", do_eval=True, do_train=False,
                                          logging_strategy=IntervalStrategy.EPOCH, save_strategy=IntervalStrategy.NO,
                                          report_to=["wandb"] if kwargs.get("use_wandb", False) else ["none"])

        trainer = Trainer(
            model, training_args, DataCollatorWithPadding(tokenizer),
            dataset['train'], dataset['test'],
            tokenizer, compute_metrics=compute_metrics)

        try:
            train = trainer.evaluate(dataset['train'], metric_key_prefix=f'{prefix}_train')
            dev = trainer.evaluate(dataset['validation'], metric_key_prefix=f'{prefix}_dev')
            test = trainer.evaluate(dataset['test'], metric_key_prefix=f'{prefix}_test')
        except ValueError:
            os.makedirs("~/dbert-error/")
            with open("~/dbert-error/dump_dataset.pickle", mode="wb") as f:
                pickle.dump(dataset, f)
            exit(-1)
        return {
            "train": train,
            "dev": dev,
            "test": test
        }

    def apply_elimination(self):
        layers = retrieve_layers(self.model)
        params = {k: v for k, v in self.elimination_params.items() if k in self.elimination.get_extra_params()}
        layers = self.elimination.discard(layers, **params)
        assign_new_layers(self.model, layers)
        self.model.config.num_hidden_layers = len(layers)

    def save(self, path: str, use_wandb: bool = False):
        pass

    def path_information(self) -> str:
        return f"{self.elimination.path_information()}/None"
