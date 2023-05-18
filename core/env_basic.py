import numpy as np
from core.exceptions import *


class Env:
    def __init__(self):
        pass

    def reset(self) -> np.ndarray:
        raise NotImplement

    def step(self, action) -> tuple:
        raise NotImplement
