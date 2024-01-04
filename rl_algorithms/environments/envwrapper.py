# main class of env
from rl_algorithms.common.core import *
from rl_algorithms.common.exceptions import *
import gymnasium as gym
import numpy as np
custom_env_list = []


class EnvWrapper:
    def __init__(self, env_id: str, logger: Logger):
        self.env_type = None
        if env_id in gym.envs.registry.keys():
            self.env_id = env_id
            self.env = gym.make(env_id)
            self.env_type = 'OpenAI GYM'
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            self.logger = logger
        else:
            # todo
            # custom environments
            pass
        self.logger('environment {env_id} from {env_type} had be built'.format(env_id=self.env_id,
                                                                               env_type=self.env_type))

    def reset(self):
        """Implement the `reset` method that initializes the environment to its initial state"""
        if self.env_type is 'OpenAI GYM':
            return self.env.reset()

    def step(self, action):
        """
        Design the `step` method to execute an action in the environment and return the new state,
        reward, and done flag.
        """
        if self.env_type is 'OpenAI GYM':
            return self.env.step(action)

    def render(self):
        """
        Include a `render` method for visualizing the environment's current state.
        """
        pass
