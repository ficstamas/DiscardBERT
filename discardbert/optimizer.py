from torch.optim import Optimizer, AdamW, SGD
from typing import Dict, Literal, Type


OptimType = Literal["adamw", "sgd"]

STR2OPTIM: Dict[OptimType, Type[Optimizer]] = {
    "adamw": AdamW,
    "sgd": SGD
}
