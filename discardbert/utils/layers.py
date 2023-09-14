from transformers import PreTrainedModel
from collections import OrderedDict
from peft import PeftModel


_encoder_names = ['encoder', 'transformer']


def retrieve_layers(model: PreTrainedModel) -> OrderedDict:
    """
    Retrieves layers from model
    :param model: Model
    :return:
    """
    if isinstance(model, PeftModel):
        return model.base_model.model.base_model.encoder.layer._modules

    for name in _encoder_names:
        if hasattr(model.base_model, name):
            return getattr(model.base_model, name).layer._modules
    raise NotImplemented('We have to define a new encoder layer name :) ')


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
        for name in _encoder_names:
            if hasattr(model.base_model, name):
                getattr(model.base_model, name).layer._modules = layers
                break
    model.config.num_hidden_layers = len(layers)
