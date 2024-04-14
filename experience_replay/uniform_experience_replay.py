import numpy as np
from abc_rl.experience_replay import *


class UniformExperienceReplay(ExperienceReplay):
    def __init__(self, memory_size: int):
        super(UniformExperienceReplay,self).__init__(memory_size)

    def sample(self, batch_size: int, dtype:np.dtype, device:torch.device):
        idx = np.arange(self.__len__())
        selected_idx = np.random.choice(idx, batch_size, replace=True)
        return self.get_items(selected_idx, dtype, device)

