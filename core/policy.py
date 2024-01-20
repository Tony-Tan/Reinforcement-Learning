import numpy as np
from utils.commons import *


class Policy:
    def __init__(self):
        pass

    def distribution(self, states: np.ndarray):
        raise PolicyNotImplement()

    def select_action(self, states: np.ndarray):
        raise PolicyNotImplement()