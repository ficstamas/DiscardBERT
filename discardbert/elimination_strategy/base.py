import abc
from collections import OrderedDict


class LayerEliminationStrategy(abc.ABC):
    @abc.abstractmethod
    def discard(self, layers: OrderedDict, *args, **kwargs) -> OrderedDict:
        pass
