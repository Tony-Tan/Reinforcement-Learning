import numpy as np
from core.env_basic import Env
from core.agent import Agent
from core.utils import *
import sys
from torch.utils.tensorboard import SummaryWriter

import cv2

class Recorder:
    def __init__(self, log_path):
        self.log_path = log_path
        self.writer = SummaryWriter(log_path)

    def __call__(self, episode_i: int, log_info: dict):
        """
        :param log_info:
        :param episode_i:
        :param log_info: {name: [value, description]}
        :return:
        """
        # print and record reward and loss
        # print('\n======================================================================')
        for key_i in log_info.keys():
            standard_info_print(key_i + ': ' + str(log_info[key_i][0]))
            self.writer.add_scalar(key_i, log_info[key_i][0], episode_i)


class Training:
    def __init__(self, env: Env, agent: Agent, max_steps: int = sys.maxsize, max_episodes: int = sys.maxsize):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.learning_step = 0
        self.learning_episode = 0

    def online_training(self):
        obs = self.env.reset()
        while self.learning_episode < self.max_episodes and \
                self.learning_step < self.max_steps:
            action = self.agent.react(obs)
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            self.agent.observe(obs, action, reward, terminated, truncated, info)
            self.agent.learn()
            self.learning_step += 1
            if terminated or truncated:
                obs = self.env.reset()
                self.learning_episode += 1
            else:
                obs = obs_next

    def offline_training(self):
        pass

