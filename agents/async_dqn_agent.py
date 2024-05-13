import torch.multiprocessing as mp
from agents.dqn_agent import *


class AsynDQNValueFunction(DQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device, logger: Logger):
        super(AsynDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                   gamma, step_c, model_saving_period, device, logger)
        self.value_nn.share_memory()
        self.target_value_nn.share_memory()


class AsyncDQNAgent(DQNAgent):
    def __init__(self,input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(AsyncDQNAgent, self).__init__(input_frame_width, input_frame_height, action_space,
                                            mini_batch_size, replay_buffer_size, mini_batch_size+1,
                                            learning_rate, step_c, model_saving_period,
                                            gamma, training_episodes, phi_channel, epsilon_max, epsilon_min,
                                            exploration_steps, device, logger)


