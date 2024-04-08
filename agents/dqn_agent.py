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


# class DQNReplayBuffer(UniformExperienceReplay):
#     def __init__(self, memory_size: int):
#         super().__init__(memory_size)
#
#     def sample(self, batch_size: int):
#         idx = np.arange(self.__len__()-1)
#         selected_idx = np.random.choice(idx, batch_size, replace=True)
#         sampled_transitions = [[] for _ in range(self.dim()+1)]
#         for idx_i in selected_idx:
#             i = 0
#             for i, data_i in enumerate(self.buffer[idx_i]):
#                 sampled_transitions[i].append(data_i)
#             sampled_transitions[i+1].append(self.buffer[idx_i + 1][0])
#         for s_i in range(len(sampled_transitions)):
#             sampled_transitions[s_i] = np.array(sampled_transitions[s_i], dtype=np.float32)
#         return sampled_transitions

def image_normalization(image_uint8: torch.Tensor) -> torch.Tensor:
    return image_uint8 / 255.0


class DQNAtariReward(RewardShaping):
    def __init__(self, skip_k_frame: int):
        super().__init__()
        self.skip_k_frame = skip_k_frame
        self.reward_cumulated = 0
        pass

    def reset(self):
        self.reward_cumulated = 0

    def __call__(self, reward, step_i: int):
        if step_i % self.skip_k_frame == 0:
            # preprocess the obs to a certain size and load it to phi
            reward_rs = self.reward_cumulated
            self.reset()
            return np.clip(reward_rs, a_min=-1,a_max=1)
        else:
            self.reward_cumulated += reward
            return None


class DQNPerceptionMapping(PerceptionMapping):
    def __init__(self, phi_channel: int, skip_k_frame: int, input_frame_width: int,
                 input_frame_height: int):
        super().__init__()
        self.phi = deque(maxlen=phi_channel)
        self.phi_channel = phi_channel
        self.skip_k_frame = skip_k_frame
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
        obs_y = cv2.cvtColor(obs, cv2.COLOR_BGR2YUV)[:, :, 0]
        # if self.last_frame_pre_process is not None:
        #     obs_y = np.maximum(self.last_frame_pre_process, img_y_channel)
        # else:
        #     obs_y = self.last_frame_pre_process = img_y_channel
        # self.last_frame_pre_process = img_y_channel
        obs_processed = cv2.resize(obs_y, (self.input_frame_width, self.input_frame_height))

        return obs_processed

    def __phi_append(self, obs: np.ndarray):
        self.phi.append(obs)

    def reset(self):
        # self.last_frame_pre_process = None
        self.phi.clear()
        for i in range(self.phi_channel):
            self.phi.append(np.zeros([self.input_frame_width, self.input_frame_width]))

    def __call__(self, state: np.ndarray, step_i: int = 0) -> np.ndarray:
        if step_i == 0:
            self.reset()
        obs = None
        if step_i % self.skip_k_frame == 0:
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
        self.__synchronize_value_nn()
        # gpt suggest that the learning rate should be schedualed
        # self.optimizer = torch.optim.RMSprop(self.value_nn.parameters(), lr=learning_rate, momentum=0.95,
        #                                      alpha=0.95, eps=0.01)
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.update_step = 0
        self.step_c = step_c
        self.model_saving_period = model_saving_period

    # def __networks_init(self):
    #     self.value_nn.to(self.device)
    #     self.target_value_nn.to(self.device)

    def __synchronize_value_nn(self):
        # print('__synchronize_value_nn')
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def update(self, samples: list):
        obs_tensor = image_normalization(torch.as_tensor(samples[0], dtype=torch.float32).to(
            self.device))
        action_tensor = torch.as_tensor(samples[1], dtype=torch.float32).to(self.device)  #
        reward_tensor = torch.as_tensor(samples[2], dtype=torch.float32).to(
            self.device)
        termination_tensor = torch.as_tensor(samples[3], dtype=torch.float32).to(self.device)  #
        truncated_tensor = torch.as_tensor(samples[4], dtype=torch.float32).to(
            self.device)
        next_obs_tensor = image_normalization(torch.as_tensor(samples[5], dtype=torch.float32).to(
            self.device))

        with torch.no_grad():
            outputs = self.target_value_nn(next_obs_tensor)
        max_next_state_value, _ = torch.max(outputs, dim=1, keepdim=True)
        # reward array
        reward_tensor.resize_as_(max_next_state_value)
        reward_tensor = torch.clamp(reward_tensor, min=-1., max=1.)
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
        outputs = self.value_nn(obs_tensor)
        obs_action_value = outputs.gather(1, actions)
        loss = torch.clip(q_value - obs_action_value, min=-1, max=1)
        # loss = F.mse_loss(q_value, obs_action_value)
        # loss = F.mse_loss(loss, torch.zeros_like(loss))
        loss = torch.mean(loss ** 2)
        # Minimize the loss
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.step_c == 0:
            self.__synchronize_value_nn()
            # self.logger.msg('synchronize target value network')
            # self.logger.tb_scalar('lr', self.lr_scheduler.get_last_lr()[0], self.update_step)
            self.logger.tb_scalar('loss', loss.item(), self.update_step)
        # if (self.update_step > 1_000_000 and self.update_step % 500_000 == 0 and
        #         self.lr_scheduler.get_last_lr()[0] > 0.00001):
        #     self.lr_scheduler.step()

    def value(self, phi_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            phi_tensor = phi_tensor.to(self.device)
            if phi_tensor.dim() == 3:
                obs_input = phi_tensor.unsqueeze(0)
            else:
                obs_input = phi_tensor
            state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            # value_of_action_list = state_action_values[0]
            return state_action_values


class DQNAgent(Agent):
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, min_update_sample_size: int, skip_k_frame: int,
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
        self.perception_mapping = DQNPerceptionMapping(phi_channel, skip_k_frame, input_frame_width, input_frame_height)
        self.reward_shaping = DQNAtariReward(skip_k_frame)
        # hyperparameters
        self.mini_batch_size = mini_batch_size
        self.skip_k_frame = skip_k_frame
        self.update_sample_size = min_update_sample_size
        self.training_episodes = training_episodes
        self.last_action = None
        self.log_avg_value = 0

    def select_action(self, obs: np.ndarray, exploration_method: Exploration = None) -> np.ndarray:
        if obs is not None:
            if isinstance(exploration_method, RandomAction):
                self.last_action = exploration_method(self.action_dim)
            else:
                obs_scaled = image_normalization(np.array(obs).astype(np.float32))
                phi_tensor = torch.from_numpy(obs_scaled)
                value_list = self.value_function.value(phi_tensor)[0]
                # self.log_avg_value += np.mean(value_list)
                if exploration_method is None:
                    self.last_action = self.exploration_method(value_list)
                    # print(value_list)
                    # print(self.last_action)
                else:
                    self.last_action = exploration_method(value_list)
                    # print('testing')
                    # print(value_list)
                    # print(self.last_action)
        return self.last_action

    # def random_action(self):
    #     self.last_action = random.randint(0, self.action_dim-1)
    #     return self.last_action

    def store(self, obs, action, reward, terminated, truncated, inf):
        if obs is not None:
            # self.memory.store([obs, action, reward, ])
            if len(self.memory) >= 1:
                self.memory[-1][-1] = obs
            self.memory.store([obs, action, reward, terminated, truncated, np.zeros_like(obs)])

    def store_termination(self):
        if len(self.memory) > 1:
            self.memory[-1][3] = True
            self.memory[-1][4] = True

    def train_step(self, step_i: int):
        if (len(self.memory) > self.update_sample_size) and (step_i % self.skip_k_frame == 0):
            samples = self.memory.sample(self.mini_batch_size)
            self.value_function.update(samples)
