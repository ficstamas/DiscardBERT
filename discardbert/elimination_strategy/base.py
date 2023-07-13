import abc
from collections import OrderedDict


class LayerEliminationStrategy(abc.ABC):
    @abc.abstractmethod
    def discard(self, layers: OrderedDict, *args, **kwargs) -> OrderedDict:
        pass

    @abc.abstractmethod
    def get_extra_params(self):
        pass

    @abc.abstractmethod
    def path_information(self) -> str:
        pass
