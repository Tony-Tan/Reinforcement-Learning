import cv2
import numpy as np
import torch.optim
import torch.nn.functional as F

from abc_rl.agent import Agent
from models.dqn_networks import DQNAtari
from abc_rl.policy import *
from abc_rl.exploration import *
from experience_replay.uniform_experience_replay import *
from abc_rl.perception_mapping import *
from abc_rl.reward_shaping import *
from exploration.epsilon_greedy import *


def image_normalization(image_uint8: torch.Tensor) -> torch.Tensor:
    # normalize the image to [-0.5,0.5]
    return image_uint8 / 255.0 - .5


class DQNAtariReward(RewardShaping):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, reward):
        # preprocess the obs to a certain size and load it to phi
        return np.clip(reward, a_min=-1, a_max=1)


class DQNPerceptionMapping(PerceptionMapping):
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
        to encode a single frame we take the maximum value for each pixel colour value over the frame being encoded
        and the previous frame. This was necessary to remove flickering that is present in games where some objects
        appear only in even frames while other objects appear only in odd frames, an artefact caused by the limited
        number of sprites Atari 2600 can display at once. Second, we then extract the Y channel, also known as
        luminance, from the RGB frame and rescale it to 84 3 84.
        :param obs: 2-d int matrix, original state of environment
        :return: 2-d float matrix, 1-channel image with size of self.down_sample_size and the value is
        converted to [-0.5,0.5]
        """
        img_y_channel = cv2.cvtColor(obs, cv2.COLOR_BGR2YUV)[:, :, 0]
        if self.last_frame_pre_process is not None:
            # take the maximum value for each pixel colour value over the
            # frame being encoded and the previous frame to remove flickering
            obs_y = np.maximum(self.last_frame_pre_process, img_y_channel)
        else:
            obs_y = self.last_frame_pre_process = img_y_channel
        self.last_frame_pre_process = img_y_channel
        obs_processed = cv2.resize(obs_y, (self.input_frame_width, self.input_frame_height))
        return obs_processed

    def __phi_append(self, obs: np.ndarray):
        self.phi.append(obs)

    def reset(self):
        # reset the phi to zero and reset the last_frame_pre_process
        self.last_frame_pre_process = None
        self.phi.clear()
        for i in range(self.phi_channel):
            self.phi.append(np.zeros([self.input_frame_width, self.input_frame_width]))

    def __call__(self, state: np.ndarray, step_i: int) -> np.ndarray:
        if step_i == 0:
            self.reset()
        # preprocess the obs to a certain size and load it to phi
        self.__phi_append(self.__pre_process(state))
        obs = np.array(self.phi)
        return obs


class DQNValueFunction(ValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device, logger: Logger):
        super(DQNValueFunction, self).__init__()
        self.logger = logger
        self.value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn.eval()
        self.__synchronize_value_nn()
        # self.optimizer = torch.optim.RMSprop(self.value_nn.parameters(), lr=learning_rate, momentum=0.95,
        #                                      alpha=0.95, eps=0.01)
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.update_step = 0
        self.step_c = step_c
        self.model_saving_period = model_saving_period

    def __synchronize_value_nn(self):
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def update(self, samples: list):
        obs_tensor = image_normalization(samples[0])
        action_tensor = samples[1]
        reward_tensor = samples[2]
        termination_tensor = samples[4]
        truncated_tensor = samples[5]
        next_obs_tensor = image_normalization(samples[3])

        with torch.no_grad():
            outputs = self.target_value_nn(next_obs_tensor)
        max_next_state_value, _ = torch.max(outputs, dim=1, keepdim=True)
        # reward array
        reward_tensor.resize_as_(max_next_state_value)
        # calculate q value
        truncated_tensor.resize_as_(max_next_state_value)
        termination_tensor.resize_as_(max_next_state_value)
        q_value = reward_tensor + self.gamma * max_next_state_value * (1 - truncated_tensor) * (1 - termination_tensor)
        # action array
        action_tensor.resize_as_(reward_tensor)
        # train the model
        q_value.resize_as_(reward_tensor)
        actions = action_tensor.long()
        self.optimizer.zero_grad()
        self.value_nn.train()
        outputs = self.value_nn(obs_tensor)
        obs_action_value = outputs.gather(1, actions)
        loss = F.mse_loss(q_value, obs_action_value)
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.step_c == 0:
            self.__synchronize_value_nn()
            self.logger.tb_scalar('loss', loss.item(), self.update_step)
            self.logger.tb_scalar('q', torch.mean(q_value), self.update_step)

    def value(self, phi_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            phi_tensor = phi_tensor.to(self.device)
            if phi_tensor.dim() == 3:
                obs_input = phi_tensor.unsqueeze(0)
            else:
                obs_input = phi_tensor
            self.value_nn.eval()
            state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            return state_action_values


class DQNAgent(Agent):
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, min_update_sample_size: int,
                 learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(DQNAgent, self).__init__(logger)
        # basic elements initialize
        self.action_dim = action_space.n
        self.value_function = DQNValueFunction(phi_channel, self.action_dim, learning_rate, gamma, step_c,
                                               model_saving_period, device, logger)
        self.exploration_method = DecayingEpsilonGreedy(epsilon_max, epsilon_min, exploration_steps)
        self.memory = UniformExperienceReplay(replay_buffer_size)
        self.perception_mapping = DQNPerceptionMapping(phi_channel, input_frame_width, input_frame_height)
        self.reward_shaping = DQNAtariReward()
        self.device = device
        # hyperparameters
        self.mini_batch_size = mini_batch_size
        self.update_sample_size = min_update_sample_size
        self.training_episodes = training_episodes

    def select_action(self, obs: np.ndarray, exploration_method: Exploration = None) -> np.ndarray:

        if isinstance(exploration_method, RandomAction):
            return exploration_method(self.action_dim)
        else:
            obs_scaled = image_normalization(np.array(obs).astype(np.float32))
            phi_tensor = torch.from_numpy(obs_scaled)
            value_list = self.value_function.value(phi_tensor)[0]
            if exploration_method is None:
                return self.exploration_method(value_list)
            else:
                return exploration_method(value_list)

    def store(self, obs, action, reward, next_obs, done, truncated):
        self.memory.store(obs, action, reward, next_obs, done, truncated)

    def train_step(self):
        if len(self.memory) > self.update_sample_size:
            samples = self.memory.sample(self.mini_batch_size, np.float32, self.device)
            self.value_function.update(samples)

