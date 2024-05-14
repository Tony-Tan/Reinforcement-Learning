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
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, learning_rate: float, step_c: int,
                 model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(AsyncDQNAgent, self).__init__(input_frame_width, input_frame_height, action_space,
                                            mini_batch_size, replay_buffer_size, mini_batch_size + 1,
                                            learning_rate, step_c, model_saving_period,
                                            gamma, training_episodes, phi_channel, epsilon_max, epsilon_min,
                                            exploration_steps, device, logger)

    def train_step(self, rank: int = 0):
        """
        Perform a training step if the memory size is larger than the update sample size.
        """
        if len(self.memory) > self.replay_start_size:
            samples = self.memory.sample(self.mini_batch_size)
            loss, q = self.value_function.update(samples)
            self.update_step += 1
            # synchronize the target value neural network with the value neural network every step_c steps
            if self.update_step % self.step_c == 0 and rank == 0:
                self.value_function.synchronize_value_nn()
                if self.logger:
                    self.logger.tb_scalar('loss', np.mean(loss), self.update_step)
                    self.logger.tb_scalar('q', np.mean(q), self.update_step)
