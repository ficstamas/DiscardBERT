from transformers import PreTrainedModel
from collections import OrderedDict
from peft import PeftModel


def retrieve_layers(model: PreTrainedModel) -> OrderedDict:
    """
    Retrieves layers from model
    :param model: Model
    :return:
    """
    if isinstance(model, PeftModel):
        return model.base_model.model.base_model.encoder.layer._modules
    return model.base_model.encoder.layer._modules


def assign_new_layers(model: PreTrainedModel, layers: OrderedDict):
    """
    Assigns new layers to the model
    :param model: Model
    :param layers: Layers
    :return:
    """
    if isinstance(model, PeftModel):
        model.base_model.model.base_model.encoder.layer._modules = layers
    else:
        model.base_model.encoder.layer._modules = layers
    model.config.num_hidden_layers = len(layers)
