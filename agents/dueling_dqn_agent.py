from agents.dqn_agent import *
from models.dqn_networks import DuelingDQNAtari


class DuelingDQNValueFunction(DQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device, logger: Logger):
        super(DuelingDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                      gamma, step_c, model_saving_period, device, logger)
        self.value_nn = DuelingDQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DuelingDQNAtari(input_channel, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)


class DuelingDQNAgent(DQNAgent):
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, min_update_sample_size: int,
                 learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(DuelingDQNAgent, self).__init__(input_frame_width, input_frame_height, action_space, mini_batch_size,
                                              replay_buffer_size, min_update_sample_size, learning_rate, step_c,
                                              model_saving_period, gamma, training_episodes, phi_channel, epsilon_max,
                                              epsilon_min, exploration_steps, device, logger)
        self.value_function = DuelingDQNValueFunction(phi_channel, action_space.n, learning_rate,
                                                      gamma, step_c, model_saving_period, device, logger)
