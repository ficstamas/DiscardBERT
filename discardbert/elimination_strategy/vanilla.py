import copy

from .base import LayerEliminationStrategy, OrderedDict


class ExactLayerEliminationStrategy(LayerEliminationStrategy):
    def discard(self, layers: OrderedDict, *args, **kwargs) -> OrderedDict:
        """
        Eliminates layers from the model based on the provided list.
        You can provide a list of layer ids in the `exact_layers` argument
        :param layers: OrderedDict containing the layers
        :param args:
        :param kwargs: exact_layers: List[int]
        :return:
        """
        exact_layers: list[int] = kwargs.get("exact_layers", ())
        new_index = 0
        new_dict = OrderedDict()
        for layer in exact_layers:
            new_dict[str(new_index)] = copy.deepcopy(layers[str(layer)])
        return new_dict

    def get_extra_params(self):
        return ["exact_layers"]

    def path_information(self, **kwargs) -> str:
        exact_layers = kwargs.get("exact_layers", ())
        exact_layers = '-'.join(exact_layers)
        return f"exact-elimination_exact-layers_{exact_layers}"


class RangeBasedLayerEliminationStrategy(ExactLayerEliminationStrategy):
    def discard(self, layers: OrderedDict, *args, **kwargs) -> OrderedDict:
        """
        Eliminates layers from the model based on the provided list.
        You can provide a list of layer ids in the `range` argument.
        ˙range˙: [start, end)
        :param layers: OrderedDict containing the layers
        :param args:
        :param kwargs: range: List[int]
        :return:
        """
        exact_layers: list[int] = kwargs.get("range", None)
        return ExactLayerEliminationStrategy.discard(
            self, layers, exact_layers=[i for i in range(exact_layers[0], exact_layers[1])]
        )

    def get_extra_params(self):
        return ["range"]

    def path_information(self, **kwargs) -> str:
        rang = [str(x) for x in kwargs.get("range", [None, ])]
        return f"range-elimination_range_{'-'.join(rang)}"
