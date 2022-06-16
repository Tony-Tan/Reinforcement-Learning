import numpy as np
from torch.utils.data import Dataset
import torch
import time
from core.exceptions import *


def print_time():
    print('Time step:' + time.strftime("%H:%M:%S", time.localtime()))


def discount_cumulate(reward_array: np.ndarray, termination_array: np.ndarray, discount_rate=0.99):
    """
    :param reward_array: nx1 float matrix
    :param discount_rate: float number
    :param termination_array: termination array, nx1 int matrix, 1 for True and 0 for False,
           discounting restart from the termination element which was 1
    :return:
    """
    step_size = len(reward_array)
    g = np.zeros(step_size, dtype=np.float32)
    g[-1] = reward_array[-1]
    for step_i in reversed(range(step_size - 1)):
        reward_i = reward_array[step_i]
        if termination_array is not None:
            if termination_array[step_i] ==1.0 :
                g[step_i] = reward_i
            else:
                g[step_i] = reward_i + discount_rate * g[step_i + 1]
        else:
            g[step_i] = reward_i + discount_rate * g[step_i + 1]
    return g.reshape((-1, 1))


class DataBuffer:
    def __init__(self, max_size: int, data_template: dict):
        self.buffer_template = data_template
        self.max_size = max_size
        self.ptr = 0
        self.buffer = {}
        for key_i in data_template.keys():
            if data_template[key_i] > 1:
                self.buffer[key_i] = np.zeros([self.max_size, data_template[key_i]], dtype=np.float32)
            else:
                self.buffer[key_i] = np.zeros([self.max_size, 1], dtype=np.float32)

    def __getitem__(self, key):
        return self.buffer[key]

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def push(self, data_list: list):
        """
        when the buffer is full, the oldest elements will be removed

        """
        data_list_idx = 0
        for key_i in self.buffer.keys():
            self.buffer[key_i][self.ptr] = data_list[data_list_idx]
            data_list_idx += 1
            if data_list_idx == len(data_list):
                break
        self.ptr = (self.ptr + 1) % self.max_size

    def insert(self, key_name: str, data: np.ndarray):
        if key_name not in self.buffer.keys():
            self.buffer[key_name] = np.zeros([self.max_size, len(data[0])], dtype=np.float32)
        ptr = self.ptr
        n = len(data)
        if n > self.max_size:
            raise OutOfRange
        for i in range(n):
            self.buffer[key_name][ptr - i - 1] = data[n-i-1]

    def clean(self):
        self.ptr = 0


class RLDataset(Dataset):
    def __init__(self, data_dic: dict, length: int):
        self.data_dict = data_dic
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        for key_i in self.data_dict.keys():
            sample[key_i] = self.data_dict[key_i][idx]
        for key_i in sample.keys():
            sample[key_i] = torch.from_numpy(sample[key_i])
        return sample

