from abc import ABC, abstractmethod

from core.commons import Logger


class RewardShaping(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...
