# import torch
# from core.exceptions import *
# import numpy as np
# import torch
# import sys
# import argparse
# from collections import deque
# import random
#
#
# class Environment:
#     def __init__(self): ...
#
#     def step(self, action: torch.Tensor) -> []:
#         return None, None, None, None
#
#     def reset(self) -> np.ndarray: ...
#
#
# class Memory:
#     """
#
#     """
#
#     def __init__(self, max_size: int = 0):
#         self.max_data_buffer_size = max_size
#         self.data_buffer = deque(maxlen=max_size)
#
#     def append(self, data):
#         self.data_buffer.append(data)
#
#     def __len__(self):
#         return len(self.data_buffer)
#
#     def __getitem__(self, item):
#         return self.data_buffer[item]
#
#     def sample(self, size: int):
#         return random.sample(self.data_buffer, size)
#
#
# class Imagination:
#     def __init__(self):
#         raise NotImplement
#
#
# class Actor:
#     def __init__(self):
#         raise NotImplement
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         raise NotImplement
#
#     def update(self, memory: Memory):
#         raise NotImplement
#
#     def __sample_from_pdf(self, samples_num: int):
#         raise NotImplement
#
#     def __sample_from_pmf(self, samples_num: int):
#         raise NotImplement
#
#     def generate_action(self, samples_num: int):
#         raise NotImplement
#
#     def pdf(self, x: torch.Tensor):
#         raise NotImplement
#
#
# class Critic:
#     def __init__(self):
#         super(Critic, self).__init__()
#         raise NotImplement
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         raise NotImplement
#
#     def update(self, *args):
#         """
#         learning parameters of neural networks based on the data in memory
#         :param memory:
#         :return:
#         """
#         raise NotImplement
#
#     def generate_action(self, *args):
#         raise NotImplement
#
#     def save(self):
#         raise NotImplement
#
#     def load(self):
#         raise NotImplement
#
#
# class Agent:
#     def __init__(self):
#         self.memory = Memory()
#
#     def reaction(self, observation):
#         raise NotImplement
#
#     def receive(self, data: list):
#         raise NotImplement
#
#     def learn(self):
#         """
#         actor learning
#         critic learning
#         a-c learning
#         :return:
#         """
#         raise NotImplement
#
#     def log_writer(self):
#         ...
#
#
# class PlayGround:
#     """
#     where agent interacted with environment and generate observation
#     """
#
#     def __init__(self, env: Environment, agent: Agent, workers_num: int = 0):
#         self.env = env
#         self.agent = agent
#         self.workers_num = workers_num
#
#     def __play_serially(self, max_steps_num: int):
#         trajectory_num = 0
#         steps_num = 0
#         observation = self.env.reset()
#         data_list = []
#         step_i = 0
#         while steps_num <= max_steps_num:
#             action = self.agent.reaction(observation)
#             new_observation, reward, is_done, _ = self.env.step(action)
#             # data_list.append([observation, action, reward, is_done, _])
#             self.agent.receive([step_i, observation, action, reward, is_done, _])
#             steps_num += 1
#             step_i += 1
#             if is_done is True:
#                 observation = self.env.reset()
#                 step_i = 0
#             else:
#                 observation = new_observation
#         return data_list
#
#     def __play_parallel(self):
#         pass
#
#     def play_rounds(self, max_steps_num: int) -> []:
#         if self.workers_num <= 1:
#             return self.__play_serially(max_steps_num)
