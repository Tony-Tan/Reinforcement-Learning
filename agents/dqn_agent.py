import cv2
import torch.optim
import torch.nn.functional as F

from rl_abc.agent import AgentOnline
from models.dqn_networks import DQNAtari
from collections import deque
from utils.commons import Logger
from environments.envwrapper import EnvWrapper
from abc.policy import *


class DQNPolicy(ValueBasedPolicy):
    def __init__(self, env: EnvWrapper, exploration_method: EpsilonGreedy, input_channel: int, mini_batch_size: int,
                 min_sample_size_update: int, learning_rate: float, gamma: float, max_training_steps: int,
                 device: str):
        # input_channel: phi_temp_size
        super().__init__(env, exploration_method)
        self.value_nn = DQNAtari(input_channel, self.action_space)
        self.target_value_nn = DQNAtari(input_channel, self.action_space)
        self.device = device  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mini_batch_size = mini_batch_size
        self.min_sample_size_update = min_sample_size_update
        self.max_training_steps = max_training_steps
        self.test_episode = 100
        self.gamma = gamma
        self.epsilon = 1
        self.optimizer = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        self.__networks_init()

    def __networks_init(self):
        self.value_nn.to(self.device)
        self.target_value_nn.to(self.device)

    def react(self, phi_np: np.ndarray, **kwargs) -> np.ndarray:
        state_action_values = self.value(phi_np, self.value_nn)
        return self.exploration(state_action_values, self.epsilon)

    def value(self, phi_np: np.ndarray, value_network: DQNAtari) -> np.ndarray:
        with torch.no_grad():
            phi = torch.as_tensor(phi_np.astype(np.float32)).to(self.device)
            if phi.dim() == 3:
                obs_input = phi.unsqueeze(0)
            else:
                obs_input = phi
            state_action_values = value_network(obs_input).cpu().detach().numpy()
            # value_of_action_list = state_action_values[0]
            return state_action_values

    def update(self, replay_buffer: ReplayBuffer):
        if len(replay_buffer) > self.min_sample_size_update:
            samples = replay_buffer.sample(self.mini_batch_size)
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
            # next state value predicted by target value networks
            max_next_state_value = []
            # with torch.no_grad():
            #     inputs = torch.from_numpy(next_obs_array).to(self.device)
            #     outputs = self.target_value_nn(inputs)
            outputs = self.value(next_obs_array, self.target_value_nn)
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
            outputs = self.value_nn(inputs)
            obs_action_value = outputs.gather(1, actions)
            loss = F.mse_loss(obs_action_value, q_value)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class DQN(AgentOnline):
    def __init__(self, env: EnvWrapper, policy:DQNPolicy, replay_buffer_size: int, skip_k_frame: int,
                 phi_temp_size: int, input_frame_width: int, input_frame_height: int,
                 init_data_size: int, gamma: float, step_c: int, model_saving_period: int,
                 device: str, save_path: str, logger: Logger):
        # agent elements settings
        super().__init__(replay_buffer_size, save_path)
        self.policy = policy
        self.input_frame_width = input_frame_width
        self.input_frame_height = input_frame_height
        # self.epsilon = 1.0
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
        # self.env = env
        # self.env_test = copy.deepcopy(env)
        self.action_dim = env.action_space.shape  # todo
        self.__phi_skf_reset()

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

    def __phi_load(self, obs: np.ndarray) -> np.ndarray:
        self.phi.append(obs)
        self.phi_np = np.array(self.phi).astype(np.float32)
        return self.phi_np

    def __phi_skf_reset(self):
        self.phi.clear()
        for i in range(self.phi_temp_size):
            self.phi.append(np.zeros([self.input_frame_width, self.input_frame_width]))
        self.skf_counter = 0
        self.skf_reward_sum = 0

    def observe(self, **kwargs):
        obs, action, reward, terminated, truncated, inf = kwargs['transition']
        if terminated or truncated:
            self.__phi_skf_reset()
        elif (self.skf_counter % self.skf) == 0:
            self.skf_reward_sum += reward
            self.replay_buffer.append([self.phi_np, action, self.skf_reward_sum, terminated, truncated,
                                       np.zeros_like(self.phi_np)])
            if len(self.replay_buffer) > 1:
                self.replay_buffer[-2][-1] = self.phi_np
            self.skf_counter += 1
            self.skf_reward_sum = 0
        else:
            self.skf_counter += 1
            self.skf_reward_sum += reward
            # self.last_episodic_reward += reward
            # self.last_episodic_steps += 1

    def react(self, **kwargs):
        self.policy.react(self.phi_np)
        return

    def load(self, map_location=torch.device('cpu')):
        # self.value_nn.load_state_dict(torch.load(self.save_path, map_location))
        pass
    def save(self):
        # save model files
        # self.logger('model saved')
        # now = int(round(time.time() * 1000))
        # now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
        # torch.save(self.value_nn.state_dict(), self.save_path + '/' + now02 + '_value.pth')
        # torch.save(self.target_value_nn.state_dict(), self.save_path + '/' + now02 + '_value_target.pth')
        pass

    def __synchronize_q_network(self):
        self.policy.target_value_nn.load_state_dict(self.policy.value_nn.state_dict())

    def learn(self):

        # total_loss += loss.item()
        self.policy.update(self.replay_buffer)
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
