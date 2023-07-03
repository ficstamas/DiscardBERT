from transformers import PreTrainedModel
from collections import OrderedDict


def retrieve_layers(model: PreTrainedModel) -> OrderedDict:
    """
    Retrieves layers from model
    :param model: Model
    :return:
    """
    return model.base_model.encoder.layer._modules


def assign_new_layers(model: PreTrainedModel, layers: OrderedDict):
    """
    Assigns new layers to the model
    :param model: Model
    :param layers: Layers
    :return:
    """
    model.base_model.encoder.layer._modules = layers
    model.config.num_hidden_layers = len(layers)
