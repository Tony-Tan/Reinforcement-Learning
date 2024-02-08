import cv2
import torch.optim
import torch.nn.functional as F

from abc_rl.agent import Agent
from models.dqn_networks import DQNAtari
from collections import deque
from utils.commons import Logger
from environments.envwrapper import EnvWrapper
from abc_rl.policy import *
from abc_rl.exploration import *
from abc_rl.experience_replay import *
from abc_rl.perception_mapping import *


class DQNValueFunction(ValueFunction):
    def __init__(self, input_channel: int, action_space, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device):
        super(DQNValueFunction, self).__init__()
        self.value_nn = DQNAtari(input_channel, action_space)
        self.target_value_nn = DQNAtari(input_channel, action_space)
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.update_step = 0
        self.step_c = step_c
        self.model_saving_period = model_saving_period

    def __networks_init(self):
        self.value_nn.to(self.device)
        self.target_value_nn.to(self.device)

    def __synchronize_value_nn(self):
        self.value_nn.load_state_dict(self.target_value_nn.state_dict())

    def update(self, samples: np.ndarray):
        obs_array = samples[:, 0]  # np.array(obs_array)
        action_array = samples[:, 1]  #
        reward_array = samples[:, 2]  # np.array(reward_array).astype(np.float32)
        is_done_array = samples[:, 3]  # np.array(is_done_array).astype(np.float32)
        next_obs_array = samples[:, 5]  # np.array(next_obs_array)
        # next state value predicted by target value networks
        max_next_state_value = []
        outputs = self.target_value_nn(next_obs_array)
        _, predictions = np.max(outputs, 1)
        # predictions = predictions.cpu().numpy()
        for p_i in range(len(predictions)):
            max_next_state_value.append(outputs[p_i][predictions[p_i]])
        max_next_state_value = np.array(max_next_state_value).astype(np.float32)
        max_next_state_value = (1.0 - is_done_array) * max_next_state_value
        # reward array
        reward_array = torch.from_numpy(reward_array)
        reward_array = torch.clamp(reward_array, min=-1., max=1.)
        # calculate q value
        q_value = reward_array + self.gamma * max_next_state_value
        # action array
        action_array = torch.Tensor(action_array).long()
        # train the model
        inputs = torch.from_numpy(obs_array).to(self.device)
        q_value = q_value.to(self.device).view(-1, 1)

        actions = action_array.to(self.device)
        # zero the parameter gradients
        # forward + backward + optimize
        outputs = self.target_value_nn(inputs)
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
            phi_tensor = torch.as_tensor(phi.astype(np.float32)).to(self.device)
            if phi_tensor.dim() == 3:
                obs_input = phi_tensor.unsqueeze(0)
            else:
                obs_input = phi_tensor
            state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            # value_of_action_list = state_action_values[0]
            return state_action_values


class SkipKFramesPhi(PerceptionMapping):
    def __init__(self, phi_channel: int, skip_k_frame: int, input_frame_width: int, input_frame_height: int):
        super().__init__()
        self.phi_channel = phi_channel
        self.phi = deque(maxlen=phi_channel)
        self.phi_channel = phi_channel
        self.skip_k_frame = skip_k_frame
        self.input_frame_width = input_frame_width
        self.input_frame_height = input_frame_height
        # dqn elements
        self.skip_k_frame_step_counter = 0
        self.skip_k_frame_reward_sum = 0

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
        gray_img = gray_img / 128. - 1.
        return gray_img

    def map(self, state: np.ndarray) -> np.ndarray:
        pass


class DQNAgent(Agent):
    def __init__(self, env_train: EnvWrapper, env_test: EnvWrapper,
                 mini_batch_size: int, memory_size: int, min_update_sample_size: int, skip_k_frame: int,
                 learning_rate: float, input_frame_width: int, input_frame_height: int, phi_temp_size: int,
                 gamma: float, training_episodes: int, phi_channel: int, device: torch.device):
        super(DQNAgent, self).__init__()
        # basic elements initialize
        self.env_train = env_train
        self.env_test = env_test
        self.value_function = DQNValueFunction(phi_temp_size, env_train.action_space, learning_rate, gamma, device)
        self.exploration_method = DecayingEpsilonGreedy(1, 0.0001, 1)  # todo
        self.memory = UniformExerienceReplay(memory_size)

        # hyperparameters
        self.mini_batch_size = mini_batch_size
        self.update_sample_size = min_update_sample_size
        # self.learning_rate = learning_rate
        self.training_episodes = training_episodes

    def __phi_load(self, obs: np.ndarray):
        self.phi.append(obs)

    def select_action(self) -> np.ndarray:
        value_list = self.value_function.value(np.array(self.phi).astype(np.float32))
        return self.exploration_method(value_list)

    def __episode_reset(self):
        self.phi.clear()
        for i in range(self.phi_channel):
            self.phi.append(np.zeros([self.input_frame_width, self.input_frame_width]))
        self.skip_k_frame_step_counter = 0
        self.skip_k_frame_reward_sum = 0

    def store(self, obs, action, reward, terminated, truncated, inf):
        self.skip_k_frame_step_counter += 1
        if terminated or truncated:
            self.__episode_reset()
        else:

            if self.skip_k_frame_step_counter % self.skip_k_frame == 0:
                self.skip_k_frame_reward_sum += reward
                # preprocess the obs to a certain size and load it to phi
                self.__phi_load(self.__obs_pre_process(obs))
                self.memory.store([self.phi, action, 1 if self.skip_k_frame_reward_sum > 0 else 0,
                                   terminated, truncated, None])
                if len(self.memory) > 1:
                    self.memory[-2][-1] = self.phi
                self.skip_k_frame_reward_sum = 0
            else:
                self.skip_k_frame_step_counter += 1
                self.skip_k_frame_reward_sum += reward

    def learn(self):
        if len(self.memory) > self.update_sample_size:
            samples = self.memory.sample(self.mini_batch_size)
            self.value_function.update(samples)

    #
    #
    #
    #
    # def training(self):
    #     epsilon = 1
    #     obs = self.env.reset()
    #     self.__phi_reset()
    #     obs_processed = self.__obs_pre_process(obs)
    #     phi_np = self.__phi_load(obs_processed)
    #     action = self.react(phi_np)
    #     reward_kf = 0
    #     for i in range(1, self.max_training_steps + 1):
    #         obs_next, reward, terminated, truncated, info = self.env.step(action)
    #         if i % self.skf == 0:
    #             obs_next_processed = self.__obs_pre_process(obs_next)
    #             phi_next_np = self.__phi_load(obs_next_processed)
    #             self.observe([phi_np, reward_kf + reward, terminated, truncated, phi_next_np])
    #             self.learn()
    #             reward_kf = 0
    #             if terminated or truncated:
    #                 obs = self.env.reset()
    #                 obs_processed = self.__obs_pre_process(obs)
    #                 self.__phi_reset()
    #                 phi_np = self.__phi_load(obs_processed)
    #             else:
    #                 phi_np = phi_next_np
    #             if i % self.model_saving_period == 0:
    #                 self.test()
    #
    #             epsilon = 1 - i * 0.0000009
    #             action = self.react(phi_np)
    #         else:
    #             if terminated or truncated:
    #                 obs = self.env.reset()
    #                 obs_processed = self.__obs_pre_process(obs)
    #                 self.__phi_reset()
    #                 phi_np = self.__phi_load(obs_processed)
    #                 action = self.react(phi_np)
    #                 reward_kf = 0
    #             else:
    #                 reward_kf += reward
    #
    # def test(self):
    #     self.__phi_reset()
    #     obs = self.env_test.reset()
    #     obs_processed = self.__obs_pre_process(obs)
    #     phi_np = self.__phi_load(obs_processed)
    #     episode_num = 0
    #     reward_sum_list = []
    #     reward_sum = 0
    #     while episode_num < self.test_episode:
    #         action = self.react(phi_np)
    #         obs_next, reward, terminated, truncated, info = self.env_test.step(action)
    #         reward_sum += reward
    #         obs_next_processed = self.__obs_pre_process(obs_next)
    #         phi_next_np = self.__phi_load(obs_next_processed)
    #         if terminated or truncated:
    #             obs = self.env_test.reset()
    #             obs_processed = self.__obs_pre_process(obs)
    #             self.__phi_reset()
    #             phi_np = self.__phi_load(obs_processed)
    #             episode_num += 1
    #         else:
    #             phi_np = phi_next_np
