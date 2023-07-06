from .simple import Simple, Training
from typing import Dict, Literal, Type


TrainingType = Literal["simple", "recursive"]

STR2TRAINING: Dict[str, Type[Training]] = {
    "simple": Simple
}
