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
    def __init__(self, worker_num: int, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int,  learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(AsyncDQNAgent, self).__init__(input_frame_width, input_frame_height, action_space,
                                            mini_batch_size, mini_batch_size, 0,
                                            learning_rate, step_c, model_saving_period,
                                            gamma, training_episodes, phi_channel, epsilon_max, epsilon_min,
                                            exploration_steps, device, logger)
        self.memory = [UniformExperienceReplay(mini_batch_size) for _ in range(worker_num)]

    def store(self,  obs, action, reward, next_obs, done, truncated, worker_id:int = 0):
        """
        Store the given parameters in the memory.

        :param obs: Observation
        :param action: Action
        :param reward: Reward
        :param next_obs: Next observation
        :param done: Done flag
        :param truncated: Truncated flag
        """
        self.memory[worker_id].store(obs, np.array(action), np.array(reward), next_obs, np.array(done), np.array(truncated))

    def train_step(self, worker_id: int = None)->bool:
            """
            Perform a training step if the memory size is larger than the update sample size.
            """
            memory = self.memory[worker_id]
            if len(memory) > self.mini_batch_size:
                samples = memory.sample(self.mini_batch_size)
                self.value_function.update(samples)
                return True
            return False
