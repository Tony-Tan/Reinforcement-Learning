import numpy as np
from core.exceptions import *


class Env:
    def __init__(self):
        pass

    def reset(self) -> np.ndarray:
        """
        :return: obs
        """
        raise NotImplement

    def step(self, action: np.ndarray) -> tuple:
        """

        :param action:
        :return:
        """
        raise NotImplement
