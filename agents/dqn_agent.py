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


class DQNReplayBuffer(UniformExperienceReplay):
    def __init__(self, memory_size: int):
        super().__init__(memory_size)

    def sample(self, batch_size: int):
        idx = np.arange(self.__len__()-1)
        selected_idx = np.random.choice(idx, batch_size, replace=True)
        sampled_transitions = [[] for _ in range(self.dim()+1)]
        for idx_i in selected_idx:
            i = 0
            for i, data_i in enumerate(self.buffer[idx_i]):
                sampled_transitions[i].append(data_i)
            sampled_transitions[i+1].append(self.buffer[idx_i + 1][0])
        for s_i in range(len(sampled_transitions)):
            sampled_transitions[s_i] = np.array(sampled_transitions[s_i], dtype=np.float32)
        return sampled_transitions


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
            return 1 if reward_rs != 0 else 0
        else:
            self.reward_cumulated += reward
        return None


class DQNPerceptionMapping(PerceptionMapping):
    def __init__(self, phi_channel: int, skip_k_frame: int, input_frame_width: int,
                 input_frame_height: int):
        super().__init__()
        self.phi_channel = phi_channel
        self.phi = deque(maxlen=phi_channel)
        self.phi_channel = phi_channel
        self.skip_k_frame = skip_k_frame
        self.input_frame_width = input_frame_width
        self.input_frame_height = input_frame_height

    def __pre_process(self, obs: np.ndarray):
        """
        :param obs: 2-d int matrix, original state of environment
        :return: 2-d float matrix, 1-channel image with size of self.down_sample_size
                 and the value is converted to [-0.5,0.5]
        """
        image = np.array(obs)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (self.input_frame_width, self.input_frame_height))
        gray_img = gray_img[self.input_frame_height - self.input_frame_width: self.input_frame_height,
                   0: self.input_frame_width]
        # gray_img = gray_img / 128. - 1.
        return gray_img

    def __phi_append(self, obs: np.ndarray):
        self.phi.append(obs)

    def reset(self):
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
            self.skip_k_frame_reward_sum = 0
        return obs


class DQNValueFunction(ValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device):
        super(DQNValueFunction, self).__init__()
        self.value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.target_value_nn = DQNAtari(input_channel, action_dim).to(device)
        self.optimizer = torch.optim.RMSprop(self.value_nn.parameters(), lr=learning_rate, momentum=0.95)
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
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def update(self, samples: list):
        obs_tensor = torch.as_tensor(samples[0], dtype=torch.float32).to(self.device)/128. - 1  # np.array(obs_array)
        action_tensor = torch.as_tensor(samples[1], dtype=torch.float32).to(self.device)  #
        reward_tensor = torch.as_tensor(samples[2], dtype=torch.float32).to(self.device)  # np.array(reward_array).astype(np.float32)
        is_done_tensor = torch.as_tensor(samples[3], dtype=torch.float32).to(self.device)  # np.array(is_done_array).astype(np.float32)
        next_obs_tensor = torch.as_tensor(samples[5], dtype=torch.float32).to(self.device)/128 - 1  # np.array(next_obs_array)
        # next state value predicted by target value networks
        # max_next_state_value = []
        outputs = self.target_value_nn(next_obs_tensor)
        max_next_state_value, _ = torch.max(outputs, dim=1, keepdim=True)
        # predictions = predictions.cpu().numpy()
        # for p_i in range(len(predictions)):
        #     max_next_state_value.append(outputs[p_i][predictions[p_i]])
        # max_next_state_value = np.array(max_next_state_value).astype(np.float32)
        is_done_tensor.resize_as_(max_next_state_value)
        max_next_state_value = ((1.0 - is_done_tensor) * max_next_state_value)
        # reward array
        reward_tensor.resize_as_(max_next_state_value)
        reward_tensor = torch.clamp(reward_tensor, min=-1., max=1.)
        # calculate q value
        q_value = reward_tensor + self.gamma * max_next_state_value
        # action array
        action_tensor.resize_as_(reward_tensor)
        # train the model
        q_value = q_value.view(-1, 1)
        actions = action_tensor.long()
        # zero the parameter gradients
        # forward + backward + optimize
        outputs = self.value_nn(obs_tensor)
        obs_action_value = outputs.gather(1, actions)
        loss = F.mse_loss(obs_action_value, q_value)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.step_c == 0:
            self.__synchronize_value_nn()

    def value(self, phi: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            phi_tensor = torch.as_tensor(phi, dtype=torch.float32).to(self.device)
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
                 gamma: float, training_episodes: int, phi_channel: int, device: torch.device):
        super(DQNAgent, self).__init__()
        # basic elements initialize
        self.value_function = DQNValueFunction(phi_channel, action_space.n, learning_rate, gamma, step_c,
                                               model_saving_period, device)
        # 1,000,000 from the paper
        self.exploration_method = DecayingEpsilonGreedy(1, 0.1, 1000000)

        self.memory = DQNReplayBuffer(replay_buffer_size)
        # self.memory = UniformExperienceReplay(replay_buffer_size)
        self.perception_mapping = DQNPerceptionMapping(phi_channel, skip_k_frame, input_frame_width, input_frame_height)
        self.reward_shaping = DQNAtariReward(skip_k_frame)
        # hyperparameters
        self.mini_batch_size = mini_batch_size
        self.skip_k_frame = skip_k_frame
        self.update_sample_size = min_update_sample_size
        # self.learning_rate = learning_rate
        self.training_episodes = training_episodes
        self.last_action = None
        self.last_max_value = 0

    def select_action(self, obs: np.ndarray, exploration_method: EpsilonGreedy = None) -> np.ndarray:
        if obs is not None:
            obs_scaled = np.array(obs).astype(np.float32) / 128. - 1.
            value_list = self.value_function.value(obs_scaled)[0]
            self.last_max_value = max(self.last_max_value, max(value_list))
            if exploration_method is None:
                self.last_action = self.exploration_method(value_list)
            else:
                self.last_action = exploration_method(value_list)
        return self.last_action

    def store(self, obs, action, reward, terminated, truncated, inf):
        if obs is not None:
            self.memory.store([obs, action, reward, terminated, truncated])
            # self.memory.store([obs, action, reward, terminated, truncated, np.zeros_like(obs)])
            # if len(self.memory) > 1:
            #     self.memory[-1][-1] = obs

    def train_step(self, step_i=0):
        if (len(self.memory) > self.update_sample_size) and (step_i % self.skip_k_frame == 0):
            samples = self.memory.sample(self.mini_batch_size)
            self.value_function.update(samples)
