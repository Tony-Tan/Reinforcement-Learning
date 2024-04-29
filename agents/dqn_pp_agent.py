import gc
from agents.dqn_agent import *
from experience_replay.proportional_prioritization import *


class DQNPPAgent(DQNAgent):
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space: int, mini_batch_size: int,
                 replay_buffer_size: int, replay_start_size: int, learning_rate: float, step_c: int,
                 agent_saving_period: int, gamma: float, training_steps: int, phi_channel: int, epsilon_max: float,
                 epsilon_min: float, exploration_steps: int, device: torch.device, logger: Logger, alpha: float = 0.6,
                 beta: float = 0.4):
        """
        Initialize the DQN agent with proportional prioritization.

        :param input_frame_width: The width of the input frame
        :param input_frame_height: The height of the input frame
        :param action_space: The action space of the environment
        :param mini_batch_size: The mini batch size
        :param replay_buffer_size: The size of the replay buffer
        :param replay_start_size: The start size of the replay buffer
        :param learning_rate: The learning rate
        :param step_c: The number of steps to update the target network
        :param agent_saving_period: The period of saving the agent
        :param gamma: The discount factor
        :param training_steps: The number of training steps
        :param phi_channel: The number of channels in the input frame
        :param epsilon_max: The maximum epsilon value
        :param epsilon_min: The minimum epsilon value
        :param exploration_steps: The number of exploration steps
        :param device: The device for training
        :param logger: The logger for recording training information
        :param alpha: The alpha value for proportional prioritization
        :param beta: The beta value for proportional prioritization
        """
        super(DQNPPAgent, self).__init__(input_frame_width, input_frame_height, action_space, mini_batch_size,
                                         replay_buffer_size, replay_start_size, learning_rate, step_c,
                                         agent_saving_period, gamma, training_steps, phi_channel, epsilon_max,
                                         epsilon_min, exploration_steps, device, logger)
        del self.memory
        gc.collect()
        self.memory = ProportionalPrioritization(replay_buffer_size, alpha, beta)


    def train_step(self):
        """
        Perform a training step if the memory size is larger than the update sample size.
        """
        if len(self.memory) > self.replay_start_size:
            samples, w, idx = self.memory.sample(self.mini_batch_size)
            td_err = self.value_function.update(samples, w)
            self.memory.p[idx] = td_err.reshape(1, -1) + 1e-10
