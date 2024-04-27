import random

from agents.dqn_agent import *


class DoubleDQNValueFunction(DQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device, logger: Logger):
        super(DoubleDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                     gamma, step_c, model_saving_period, device, logger)

    def max_state_value(self, obs_tensor):
        with torch.no_grad():
            outputs_tnn = self.target_value_nn(obs_tensor)
            outputs_nn = self.value_nn(obs_tensor)
        _, greedy_actions = torch.max(outputs_nn, dim=1, keepdim=True)
        msv = outputs_tnn.gather(1, greedy_actions)
        return msv


class DoubleDQNAgent(DQNAgent):
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, min_update_sample_size: int,
                 learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(DoubleDQNAgent, self).__init__(input_frame_width, input_frame_height, action_space, mini_batch_size,
                                             replay_buffer_size, min_update_sample_size, learning_rate, step_c,
                                             model_saving_period, gamma, training_episodes, phi_channel, epsilon_max,
                                             epsilon_min, exploration_steps, device, logger)
        self.value_function = DoubleDQNValueFunction(phi_channel, action_space.n, learning_rate,
                                                     gamma, step_c, model_saving_period, device, logger)


