# main class of env
import gymnasium as gym
from gymnasium import envs
from gymnasium.wrappers import AtariPreprocessing
from utils.commons import *

custom_env_list = []


class EnvError(Exception):
    def __init__(self, error_inf):
        self.error_inf = error_inf

    def __str__(self):
        return 'Environment error: ' + self.error_inf


class EnvWrapper:
    def __init__(self, env_id: str, frame_skip: int = 1, logger: Logger = None, **kwargs):
        self.env_type = None
        self.logger = logger
        if env_id in gym.envs.registry.keys():
            if 'ALE' in env_id:
                self.env_id = env_id
                self.env = gym.make(env_id, repeat_action_probability=0, frameskip=1, render_mode=None)
                self.screen_size = kwargs['screen_size'] if 'screen_size' in kwargs.keys() else 84
                self.env = AtariPreprocessing(self.env,  screen_size=self.screen_size,frame_skip=frame_skip,
                                              grayscale_obs=True, scale_obs=False)
                self.env_type = 'Atari'
                self.action_space = self.env.action_space
                self.state_space = self.env.observation_space
                if self.logger:
                    self.logger.msg(f'env id: {env_id} |repeat_action_probability: 0 |render_mode: None')
                    self.logger.msg(f'frame_skip: {frame_skip} |screen_size: {self.screen_size} |'
                                    f'grayscale_obs:{True} |scale_obs:{False}')
        else:
            raise EnvError('not exist env_id')

        # self.logger('EnvWrapper init: {env_id} from {env_type} had be built'.
        #             format(env_id=self.env_id, env_type=self.env_type))

    def reset(self):
        """Implement the `reset` method that initializes the environment to its initial state"""
        if self.env_type == 'OpenAI GYM' or self.env_type == 'Atari':
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
