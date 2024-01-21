import copy
import os
import cv2
import torch.optim
import time
import torch.nn.functional as F

from online_algorithms.agents.base_agent import AgentOnline
from online_algorithms.models.dqn_networks import DQNAtari
from collections import deque
from utils.commons import Logger
from environments.envwrapper import EnvWrapper


class DQN(AgentOnline):
    def __init__(self, env: EnvWrapper, phi_temp_size: int,
                 replay_buffer_size: int, skip_k_frame: int, mini_batch_size: int, learning_rate: float,
                 input_frame_width: int, input_frame_height: int, init_data_size: int, max_training_steps: int,
                 gamma: float, step_c: int, model_saving_period: int, device: str,
                 save_path: str, logger: Logger):
        # agent elements settings
        super().__init__(replay_buffer_size, save_path)
        self.input_frame_width = input_frame_width
        self.input_frame_height = input_frame_height
        self.epsilon = 1.0
        self.phi_temp_size = phi_temp_size
        self.phi = deque(maxlen=phi_temp_size)
        self.phi_np = None
        self.skf = skip_k_frame
        self.skf_reward_sum = 0
        self.update_steps = 1
        self.min_sample_size_necessary = init_data_size
        self.gamma = gamma
        self.step_c = step_c
        self.model_saving_period = model_saving_period
        # interaction with environment
        self.env = env
        self.env_test = copy.deepcopy(env)
        self.action_dim = env.action_space.shape  # todo
        # Networks settings
        self.value_nn = DQNAtari(phi_temp_size, self.action_dim)
        self.target_value_nn = DQNAtari(phi_temp_size, self.action_dim)
        self.device = device  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mini_batch_size = mini_batch_size
        self.max_training_steps = max_training_steps
        self.test_episode = 100
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        self.value_nn.to(self.device)
        self.target_value_nn.to(self.device)
        # make save folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.__phi_reset()

    def __obs_pre_process(self, obs: np.ndarray):
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

    def __phi_load(self, obs: np.ndarray)->np.ndarray:
        self.phi.append(obs)
        self.phi_np = np.array(self.phi).astype(np.float32)
        return self.phi_np

    def __phi_reset(self):
        self.phi.clear()
        for i in range(self.phi_temp_size):
            self.phi.append(np.zeros([self.input_frame_width, self.input_frame_width]))

    # def observe(self, obs, action, reward, terminated, truncated, inf, save_obs=True):
    #     if terminated or truncated:
    #         self.phi_reset()
    #         self.skf_counter = 0
    #         self.skf_reward_sum = 0
    #         # recorder
    #         # self.last_10_episodic_reward.append(self.last_episodic_reward)
    #         # self.last_10_episodic_steps.append(self.last_episodic_steps)
    #         # self.last_episodic_reward = 0
    #         # self.last_episodic_steps = 0
    #
    #     elif (self.skf_counter % self.skf) == 0 and save_obs:
    #         self.skf_reward_sum += reward
    #         self.replay_buffer.append([self.phi_np, action, self.skf_reward_sum, terminated, truncated,
    #                                    np.zeros_like(self.phi_np)])
    #         if len(self.memory) > 1:
    #             self.memory[-2][-1] = self.phi_np
    #         self.skf_counter += 1
    #         self.skf_reward_sum = 0
    #         # self.last_episodic_reward += reward
    #         # self.last_episodic_steps += 1
    #     else:
    #         self.skf_counter += 1
    #         self.skf_reward_sum += reward
    #         # self.last_episodic_reward += reward
    #         # self.last_episodic_steps += 1
    # def generate_action(self, phi_t: torch.Tensor, epsilon: float = None):

    def react(self, states: np.ndarray):
        with torch.no_grad():
            phi = torch.as_tensor(self.phi_np.astype(np.float32)).to(self.device)
            obs_input = phi.unsqueeze(0)
            state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            value_of_action_list = state_action_values[0]
            return epsilon_greedy(value_of_action_list, epsilon)

    def load(self, map_location=torch.device('cpu')):
        self.value_nn.load_state_dict(torch.load(self.save_path, map_location))

    def save(self):
        # save model files
        self.logger('model saved')
        now = int(round(time.time() * 1000))
        now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
        torch.save(self.value_nn.state_dict(), self.save_path + '/' + now02 + '_value.pth')
        torch.save(self.target_value_nn.state_dict(), self.save_path + '/' + now02 + '_value_target.pth')

    def __synchronize_q_network(self):
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def learn(self):
        if len(self.replay_buffer) > self.min_sample_size_necessary:
            samples = random.sample(self.replay_buffer, self.mini_batch_size)
            obs_array = []
            action_array = []
            # next_obs_array = []
            # reward_array = []
            # is_done_array = []
            samples = np.array(samples)
            # for sample_i in samples:
            #     # obs, action, self.skf_reward_sum,
            #     # terminated, truncated, next_obs
            #     obs_array.append(sample_i[0])
            #     action_array.append([sample_i[1]])
            #     reward_array.append(sample_i[2])
            #     is_done_array.append(sample_i[3])
            #     next_obs_array.append(sample_i[5])
            obs_array = samples[:, 0]  # np.array(obs_array)
            action_array = samples[:, 1]  #
            reward_array = samples[:, 2]  # np.array(reward_array).astype(np.float32)
            is_done_array = samples[:, 3]  # np.array(is_done_array).astype(np.float32)
            next_obs_array = samples[:, 5]  # np.array(next_obs_array)

            max_next_state_value = []
            with torch.no_grad():
                inputs = torch.from_numpy(next_obs_array).to(self.device)
                outputs = self.target_value_nn(inputs)
                _, predictions = torch.max(outputs, 1)
                outputs = outputs.cpu().numpy()
                predictions = predictions.cpu().numpy()
                for p_i in range(len(predictions)):
                    max_next_state_value.append(outputs[p_i][predictions[p_i]])
            max_next_state_value = np.array(max_next_state_value).astype(np.float32)
            max_next_state_value = (1.0 - is_done_array) * max_next_state_value
            reward_array = torch.from_numpy(reward_array)
            reward_array = torch.clamp(reward_array, min=-1., max=1.)
            q_value = reward_array + self.gamma * max_next_state_value

            action_array = torch.Tensor(action_array).long()
            # train the model
            inputs = torch.from_numpy(obs_array).to(self.device)
            q_value = q_value.to(self.device).view(-1, 1)

            actions = action_array.to(self.device)
            # zero the parameter gradients
            # forward + backward + optimize
            outputs = self.value_nn(inputs)
            obs_action_value = outputs.gather(1, actions)
            loss = F.mse_loss(obs_action_value, q_value)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # total_loss += loss.item()
            self.update_steps += 1
            if self.update_steps % self.step_c == 0:
                self.__synchronize_q_network()
            if self.update_steps % self.model_saving_period == 0:
                self.save()
            # self.epsilon = 1. - self.update_steps * 0.0000009
            # self.epsilon = max(args.epsilon_min, self.epsilon)

    def training(self):
        epsilon = 1
        obs = self.env.reset()
        self.__phi_reset()
        obs_processed = self.__obs_pre_process(obs)
        phi_np = self.__phi_load(obs_processed)
        action = self.react(phi_np)
        reward_kf = 0
        for i in range(1, self.max_training_steps + 1):
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            if i % self.skf == 0:
                obs_next_processed = self.__obs_pre_process(obs_next)
                phi_next_np = self.__phi_load(obs_next_processed)
                self.observe([phi_np, reward_kf + reward, terminated, truncated, phi_next_np])
                self.learn()
                reward_kf = 0
                if terminated or truncated:
                    obs = self.env.reset()
                    obs_processed = self.__obs_pre_process(obs)
                    self.__phi_reset()
                    phi_np = self.__phi_load(obs_processed)
                else:
                    phi_np = phi_next_np
                if i % self.model_saving_period == 0:
                    self.test()

                epsilon = 1 - i * 0.0000009
                action = self.react(phi_np)
            else:
                if terminated or truncated:
                    obs = self.env.reset()
                    obs_processed = self.__obs_pre_process(obs)
                    self.__phi_reset()
                    phi_np = self.__phi_load(obs_processed)
                    action = self.react(phi_np)
                    reward_kf = 0
                else:
                    reward_kf += reward


    def test(self):
        self.__phi_reset()
        obs = self.env_test.reset()
        obs_processed = self.__obs_pre_process(obs)
        phi_np = self.__phi_load(obs_processed)
        episode_num = 0
        reward_sum_list = []
        reward_sum = 0
        while episode_num < self.test_episode:
            action = self.react(phi_np)
            obs_next, reward, terminated, truncated, info = self.env_test.step(action)
            reward_sum += reward
            obs_next_processed = self.__obs_pre_process(obs_next)
            phi_next_np = self.__phi_load(obs_next_processed)
            if terminated or truncated:
                obs = self.env_test.reset()
                obs_processed = self.__obs_pre_process(obs)
                self.__phi_reset()
                phi_np = self.__phi_load(obs_processed)
                episode_num += 1
            else:
                phi_np = phi_next_np

