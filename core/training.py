import copy

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
    def __init__(self, env: Env, agent: Agent,
                 max_steps: int = sys.maxsize, max_episodes: int = sys.maxsize,
                 test_period_steps: int = 4000, test_episodes: int = 10,
                 log_path: str = './'):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.learning_steps = 0
        self.learning_episodes = 0
        self.test_period_steps = test_period_steps
        self.test_episodes = test_episodes
        self.recorder = Recorder(log_path)

    def test(self):
        reward_np = np.zeros(self.test_episodes)
        steps_np = np.zeros(self.test_episodes)
        for i in range(self.test_episodes):
            obs = self.env.reset()
            terminated, truncated = False, False
            while (not terminated) and (not truncated):
                action = self.agent.react(obs, testing=True)
                obs_next, reward, terminated, truncated, info = self.env.step(action)
                self.agent.observe(obs, action, reward, terminated, truncated, info, False)
                obs = obs_next
                reward_np[i] += reward
                steps_np[i] += 1
        self.recorder(self.learning_steps, {'reward': [np.mean(reward_np), 'reward'],
                                            'steps': [np.mean(steps_np), 'steps']})

    def online_training(self):
        obs = self.env.reset()
        while self.learning_episodes < self.max_episodes and \
                self.learning_steps < self.max_steps:
            action = self.agent.react(obs)
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            self.agent.observe(obs, action, reward, terminated, truncated, info)
            self.agent.learn()
            self.learning_steps += 1
            if terminated or truncated:
                obs = self.env.reset()
                self.learning_episodes += 1
            else:
                obs = obs_next
            if self.learning_steps % self.test_period_steps == 0:
                self.test()

    def offline_training(self):
        pass
