import abc


class Metrics(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_metrics(self, p):
        pass
