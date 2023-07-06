from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, PreTrainedModel
from typing import Literal, Dict


ModelType = Literal["sequence", "token"]


STR2MODEL_TYPE: Dict[ModelType, PreTrainedModel] = {
    "sequence": AutoModelForSequenceClassification,
    "token": AutoModelForTokenClassification
}
