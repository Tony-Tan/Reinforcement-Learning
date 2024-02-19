# main class of env
import gymnasium as gym
from gymnasium import envs
from utils.commons import *

custom_env_list = []


class EnvError(Exception):
    def __init__(self, error_inf):
        self.error_inf = error_inf

    def __str__(self):
        return 'Environment error: ' + self.error_inf


class EnvWrapper:
    def __init__(self, env_id: str, logger: Logger):
        self.env_type = None
        if env_id in gym.envs.registry.keys():
            self.env_id = env_id
            self.env = gym.make(env_id)
            self.env_type = 'OpenAI GYM'
            self.action_space = self.env.action_space
            self.state_space = self.env.observation_space
            self.logger = logger
        else:
            raise EnvError('not exist env_id')

        self.logger('EnvWrapper init: {env_id} from {env_type} had be built'.
                    format(env_id=self.env_id, env_type=self.env_type))

    def reset(self):
        """Implement the `reset` method that initializes the environment to its initial state"""
        if self.env_type == 'OpenAI GYM':
            return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        """
        Include a `render` method for visualizing the environment's current state.
        """
        pass


if __name__ == '__main__':
    for key_i in envs.registry.keys():
        print(key_i)
