import cv2
import numpy as np
import torch.optim
import torch.nn.functional as F
from collections import deque
from abc_rl.agent import Agent
from models.dqn_networks import DQNAtari
from abc_rl.policy import *
from abc_rl.exploration import *
from experience_replay.uniform_experience_replay import *
from abc_rl.perception_mapping import *
from abc_rl.reward_shaping import *
from exploration.epsilon_greedy import *


def image_normalization(image_uint8):
    """
    Normalize the image to [-0.5,0.5]

    :param image_uint8: Input image tensor
    :return: Normalized image tensor
    """
    return image_uint8 / 255.0 - .5


class DQNAtariReward(RewardShaping):
    """
    Class for reward shaping in DQN for Atari games.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, reward):
        """
        Preprocess the reward to clip it between -1 and 1.

        :param reward: Input reward
        :return: Clipped reward
        """
        return np.clip(reward, a_min=-1, a_max=1)


class DQNPerceptionMapping(PerceptionMapping):
    """
    Class for perception mapping in DQN for Atari games.
    """

    def __init__(self, phi_channel: int, input_frame_width: int,
                 input_frame_height: int):
        super().__init__()
        self.phi = deque(maxlen=phi_channel)
        self.phi_channel = phi_channel
        self.input_frame_width = input_frame_width
        self.input_frame_height = input_frame_height
        self.last_frame_pre_process = None

    def __pre_process(self, obs: np.ndarray):
        """
        Preprocess the observation by taking the maximum value for each pixel colour value over the frame being encoded
        and the previous frame. This is necessary to remove flickering that is present in games where some objects
        appear only in even frames while other objects appear only in odd frames, an artefact caused by the limited
        number of sprites Atari 2600 can display at once. Second, we then extract the Y channel, also known as
        luminance, from the RGB frame and rescale it to 84 3 84.

        :param obs: 2-d int matrix, original state of environment
        :return: 2-d float matrix, 1-channel image with size of self.down_sample_size and the value is
        converted to [-0.5,0.5]
        """
        img_y_channel = cv2.cvtColor(obs, cv2.COLOR_BGR2YUV)[:, :, 0]
        if self.last_frame_pre_process is not None:
            obs_y = np.maximum(self.last_frame_pre_process, img_y_channel)
        else:
            obs_y = self.last_frame_pre_process = img_y_channel
        self.last_frame_pre_process = img_y_channel
        obs_processed = cv2.resize(obs_y, (self.input_frame_width, self.input_frame_height))

        return obs_processed

    def __phi_append(self, obs: np.ndarray):
        """
        Append the observation to the phi deque.

        :param obs: Input observation
        """
        self.phi.append(obs)

    def reset(self):
        """
        Reset the phi to zero and reset the last_frame_pre_process.
        """
        self.last_frame_pre_process = None
        self.phi.clear()
        for i in range(self.phi_channel):
            self.phi.append(np.zeros([self.input_frame_width, self.input_frame_width]))

    def __call__(self, state: np.ndarray, step_i: int) -> np.ndarray:
        """
        Preprocess the state to a certain size and load it to phi.

        :param state: Input state
        :param step_i: Step index
        :return: Processed state
        """
        if step_i == 0:
            self.reset()
        self.__phi_append(self.__pre_process(state))
        obs = np.array(self.phi, dtype=np.uint8)
        return obs


class DQNValueFunction(ValueFunction):
    """
    Class for value function in DQN for Atari games.
    """

    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device, logger: Logger):
        super(DQNValueFunction, self).__init__()
        self.logger = logger
        self.value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn.eval()
        self.synchronize_value_nn()
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.value_nn.parameters(), lr=learning_rate, momentum=0.95,
        #                                      alpha=0.95, eps=0.01)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.update_step = 0
        self.step_c = step_c
        self.model_saving_period = model_saving_period

    def synchronize_value_nn(self):
        """
        Synchronize the value neural network with the target value neural network.
        """
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def max_state_value(self, obs_tensor):
        with torch.no_grad():
            obs_tensor = image_normalization(obs_tensor)
            outputs = self.target_value_nn(obs_tensor)
        msv, _ = torch.max(outputs, dim=1, keepdim=True)
        return msv

    def update(self, samples: list, weight=None):
        """
        Update the value function with the given samples.

        :param samples: Input samples
        :param weight: Importance weight for prioritized experience replay
        """
        # stream = torch.cuda.Stream(device=self.device)
        # with torch.cuda.stream(stream):
        obs_tensor = samples[0].to(self.device, non_blocking=True)
        action_tensor = samples[1].to(self.device, non_blocking=True)
        reward_tensor = samples[2].to(self.device, non_blocking=True)
        next_obs_tensor = samples[3].to(self.device, non_blocking=True)
        termination_tensor = samples[4].to(self.device, non_blocking=True)
        truncated_tensor = samples[5].to(self.device, non_blocking=True)

        # stream.synchronize()

        max_next_state_value = self.max_state_value(next_obs_tensor)
        reward_tensor.resize_as_(max_next_state_value)
        # calculate q value
        truncated_tensor.resize_as_(max_next_state_value)
        termination_tensor.resize_as_(max_next_state_value)
        q_value = reward_tensor + self.gamma * max_next_state_value * (1 - truncated_tensor) * (1 - termination_tensor)
        action_tensor.resize_as_(reward_tensor)
        q_value.resize_as_(reward_tensor)
        actions = action_tensor.long()
        self.optimizer.zero_grad()
        self.value_nn.train()
        # normalize the input image
        obs_tensor = image_normalization(obs_tensor)
        outputs = self.value_nn(obs_tensor)
        obs_action_value = outputs.gather(1, actions)
        # loss = F.mse_loss(q_value, obs_action_value)
        # Clip the difference between obs_action_value and q_value to the range of -1 to 1
        diff = obs_action_value - q_value
        if weight is not None:
            # for the prioritized experience replay
            weight = torch.as_tensor(weight, device=self.device, dtype=torch.float32).resize_as_(diff)
            diff_clipped = torch.clip(diff, -1, 1) * weight
        else:
            diff_clipped = torch.clip(diff, -1, 1)

        # Use the clipped difference for the loss calculation
        loss = F.mse_loss(diff_clipped, torch.zeros_like(diff_clipped))
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.step_c == 0:
            self.synchronize_value_nn()
            self.logger.tb_scalar('loss', loss.item(), self.update_step)
            self.logger.tb_scalar('q', torch.mean(q_value), self.update_step)
        return np.abs(diff_clipped.detach().cpu().numpy().astype(np.float32))

    def value(self, phi_tensor: torch.Tensor) -> np.ndarray:
        """
        Calculate the value of the given phi tensor.

        :param phi_tensor: Input phi tensor
        :return: Value of the phi tensor
        """
        with torch.no_grad():
            if phi_tensor.dim() == 3:
                obs_input = phi_tensor.unsqueeze(0)
            else:
                obs_input = phi_tensor
            self.value_nn.eval()
            obs_input = image_normalization(obs_input)
            state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            return state_action_values


class DQNAgent(Agent):
    """
    Class for DQN agent for Atari games.
    """

    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, replay_start_size: int,
                 learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(DQNAgent, self).__init__(logger)
        self.action_dim = action_space.n
        self.value_function = DQNValueFunction(phi_channel, self.action_dim, learning_rate, gamma, step_c,
                                               model_saving_period, device, logger)
        self.exploration_method = DecayingEpsilonGreedy(epsilon_max, epsilon_min, exploration_steps)
        self.memory = UniformExperienceReplay(replay_buffer_size)
        self.perception_mapping = DQNPerceptionMapping(phi_channel, input_frame_width, input_frame_height)
        self.reward_shaping = DQNAtariReward()
        self.device = device
        self.mini_batch_size = mini_batch_size
        self.replay_start_size = replay_start_size
        self.training_episodes = training_episodes

    def select_action(self, obs: np.ndarray, exploration_method: Exploration = None) -> np.ndarray:
        """
        Select an action based on the given observation and exploration method.

        :param obs: Input observation
        :param exploration_method: Exploration method
        :return: Selected action
        """
        if isinstance(exploration_method, RandomAction):
            return exploration_method(self.action_dim)
        else:
            # obs_scaled = image_normalization(np.array(obs).astype(np.float32))
            phi_tensor = torch.as_tensor(obs, device=self.device,dtype=torch.float32)
            value_list = self.value_function.value(phi_tensor)[0]
            if exploration_method is None:
                return self.exploration_method(value_list)
            else:
                return exploration_method(value_list)

    def store(self, obs, action, reward, next_obs, done, truncated):
        """
        Store the given parameters in the memory.

        :param obs: Observation
        :param action: Action
        :param reward: Reward
        :param next_obs: Next observation
        :param done: Done flag
        :param truncated: Truncated flag
        """
        self.memory.store(obs, np.array(action), np.array(reward), next_obs, np.array(done), np.array(truncated))

    def train_step(self):
        """
        Perform a training step if the memory size is larger than the update sample size.
        """
        if len(self.memory) > self.replay_start_size:
            samples = self.memory.sample(self.mini_batch_size)
            self.value_function.update(samples)
