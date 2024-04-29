import numpy as np
from abc_rl.experience_replay import *
import random


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, capacity: int):
        super(UniformExperienceReplay,self).__init__(capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(np.arange(self.__len__()), batch_size,  replace=False)
        return self.get_items(idx)

