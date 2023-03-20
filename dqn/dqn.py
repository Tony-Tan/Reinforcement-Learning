from core.elements import *
import gym
import cv2
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import time
import os
import torch.nn.functional as F
from core.basic import *

args_list = [
    ['--env_name', 'InvertedDoublePendulum-v2', str, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--mini_batch_size', 32, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--episodes_num', 100000, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--k_frames', 4, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--input_frame_width', 84, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--input_frame_height', 110, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--memory_length', 15000, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--init_data_size', 2000, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--phi_temp_size', 4, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--gamma', 0.99, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--model_path', './model/', str, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--log_path', './log/', str, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--learning_rate', 1e-5, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--steps_c', 100, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--epsilon_update_steps', 1000, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--epsilon_max', 1.0, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--epsilon_min', 0.001, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--epsilon_decay', 0.9999, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2']
]
args = script_args(args_list, 'dqn training arguments')


class DQNGym(Environment):
    def __init__(self, env_name):
        super().__init__()
        self.env = gym.make(env_name)
        self.action_dim = self.env.action_space.n

    def step(self, action: np.ndarray) -> []:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs


class DQNMemory(Memory):
    def __init__(self):
        super().__init__(max_size=args.memory_length)

    # def add(self, data: list):
    #     # data['obs'] = self.__pre_process_obs(data['obs'])
    #     if len(self.data_buffer) >= 1:
    #         self.data_buffer[-1][7] = data[1]
    #     self.append(data)


class CriticDQN(nn.Module):
    def __init__(self, input_channel_size, output_size):
        super(CriticDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.fc_2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 3136)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x


class AgentDQN(Agent):
    def __init__(self, action_dim: int, save_path: str):
        # basic configuration
        super().__init__()
        self.critic = CriticDQN(4, action_dim)
        # self.algorithm_version = args.algorithm_version
        # self.env = environment
        self.action_n = action_dim
        self.epsilon = args.epsilon_max
        self.episodes_num = args.episodes_num
        # self.input_frame_size = args.input_frame_size
        self.phi = deque(maxlen=args.phi_temp_size)
        self.memory = DQNMemory()
        self.update_steps = 1
        self.action_dim = action_dim
        self.value_nn = CriticDQN(args.phi_temp_size, self.action_dim)
        self.target_value_nn = CriticDQN(args.phi_temp_size, self.action_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mini_batch_size = args.mini_batch_size
        self.optimizer = optim.Adam(self.value_nn.parameters(), lr=args.learning_rate)
        self.value_nn.to(self.device)
        self.target_value_nn.to(self.device)
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.writer = SummaryWriter(self.save_path)
        self.model_path = self.save_path

    def receive(self, data: list):
        self.memory.append(data)

    def reaction(self, phi: np.ndarray):
        with torch.no_grad():
            phi = torch.as_tensor(phi.astype(np.float32)).to(self.device)
            return self.generate_action(phi)

    def __pre_process_obs(self, obs: np.ndarray):
        """
        :param obs: 2-d int matrix, original state of environment
        :return: 2-d float matrix, 1-channel image with size of self.down_sample_size
                 and the value is converted to [-0.5,0.5]
        """
        image = np.array(obs)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (args.input_frame_width, args.input_frame_height))

        gray_img = gray_img[args.input_frame_height - args.input_frame_width:args.input_frame_height,
                   0:args.input_frame_width]
        gray_img = gray_img / 255. - 0.5
        return gray_img

    def phi_concat(self, observation) -> np.ndarray:
        obs = self.__pre_process_obs(observation)
        self.phi.append(obs)
        phi = np.array(self.phi).astype(np.float32)
        return phi

    def phi_reset(self):
        self.phi.clear()
        for i in range(args.phi_temp_size):
            self.phi.append(np.zeros([args.input_frame_width, args.input_frame_width]))

    def load(self, model_path, map_location=torch.device('cpu')):
        # model_name_list = os.listdir(self.save_path)
        # model_files_num = 0
        # model_file_list = []
        # for model_name_i in model_name_list:
        #     if '.pth' in model_name_i:
        #         print('[' + str(model_files_num) + ']: ' + model_name_i)
        #         model_file_list.append(model_name_i)
        #         model_files_num += 1
        # if model_files_num > 0:
        #     print('found model files.')
        #     model_selected = input('which one would you load:' + '[' + str(model_files_num) + ']')
        #     if int(model_selected) in range(model_files_num):
        #         self.value_nn.load_state_dict(torch.load(self.model_path + model_selected))
        #         self.target_value_nn.load_state_dict(torch.load(self.model_path + model_selected))
        #     else:
        #         raise ModelFileNotFind
        # else:
        #     raise ModelFileNotFind
        self.value_nn.load_state_dict(torch.load(model_path, map_location))

    def save(self):
        # save model files
        print('model saved')
        now = int(round(time.time() * 1000))
        now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
        torch.save(self.value_nn.state_dict(), self.save_path + '/' + now02 + '_value.pth')
        torch.save(self.target_value_nn.state_dict(), self.save_path + '/' + now02 + '_value_target.pth')

    def generate_action(self, phi_t: torch.Tensor):
        obs_input = phi_t.unsqueeze(0)
        state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
        value_of_action_list = state_action_values[0]
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        if random.randint(0, 10000000) < self.epsilon * 10000000:
            return random.randint(0, self.action_dim - 1)
        else:
            return optimal_action

    def synchronize_q_network(self):
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def record_reward(self, frame_num, total_reward, epsilon, episode_i):
        # print and record reward and loss
        print("reward of episode: " + str(episode_i) + " is " + str(total_reward)
              + " and frame number is " + str(frame_num) + ' epsilon: ' + str(epsilon))
        self.writer.add_scalar('reward of episode', total_reward, episode_i)
        self.writer.add_scalar('steps of an episode', frame_num, episode_i)
        self.writer.add_scalar('epsilon', epsilon, episode_i)

    def learn(self):
        total_loss = 0
        # build label:
        if len(self.memory) > args.init_data_size:
            samples = self.memory.sample(self.mini_batch_size)
            # phi, action, reward, is_done, next_phi
            # obs_array = np.array([i[0] for i in samples])
            # action_array = [[i[1]] for i in samples]
            # next_obs_array = np.array([i[4] for i in samples])
            # reward_array = [i[2] for i in samples]
            # is_done_array = [i[3] for i in samples]
            obs_array = []
            action_array = []
            next_obs_array = []
            reward_array = []
            is_done_array = []
            for sample_i in samples:
                obs_array.append(sample_i[0])
                action_array.append([sample_i[1]])
                reward_array.append(sample_i[2])
                is_done_array.append(sample_i[3])
                next_obs_array.append(sample_i[4])
            obs_array = np.array(obs_array)
            next_obs_array = np.array(next_obs_array)
            is_done_array = np.array(is_done_array).astype(np.float32)
            reward_array = np.array(reward_array).astype(np.float32)

            max_next_state_value = []
            with torch.no_grad():
                inputs = torch.from_numpy(next_obs_array).to(self.device)
                outputs = self.target_value_nn(inputs)
                _, predictions = torch.max(outputs, 1)
                outputs = outputs.cpu().numpy()
                predictions = predictions.cpu().numpy()
                for p_i in range(len(predictions)):
                    max_next_state_value.append(outputs[p_i][predictions[p_i]])
            # for i in range(self.mini_batch_size):
            #     if is_done_array[i]:
            #         max_next_state_value[i] = 0
            max_next_state_value = np.array(max_next_state_value).astype(np.float32)
            max_next_state_value = (1.0 - is_done_array) * max_next_state_value
            reward_array = torch.from_numpy(reward_array)
            reward_array = torch.clamp(reward_array, min=-1., max=1.)
            q_value = reward_array + args.gamma * max_next_state_value
            # reward_array = reward_array.astype(np.float32)

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
            if self.update_steps % args.steps_c == 0:
                self.synchronize_q_network()
            if self.update_steps % 1000 == 0:
                print("\t\t" + str(self.update_steps) + "  steps, loss: " + str(loss.item()))

            self.epsilon = 1. - self.update_steps * 0.00000009
            self.epsilon = max(args.epsilon_min, self.epsilon)
            # record
            # self.writer.add_scalar('train/loss', total_loss)


class DQNPlayGround(PlayGround):
    def __init__(self, env: Environment, agent: AgentDQN):
        super().__init__(env, agent)
        self.agent = agent
        self.episode_num = 0
        self.skip_k_frame = args.k_frames
        self.agent.phi_reset()
        obs_raw = self.env.reset()
        self.phi_np = self.agent.phi_concat(obs_raw)
        self.reward_recorder = 0
        self.steps_recorder = 0
        self.last_episodic_reward = 0
        self.last_episodic_steps = 0

    def __play_serially(self, max_steps_num: int, display=False):
        steps_num = 0
        while steps_num < max_steps_num:
            action = self.agent.reaction(self.phi_np)
            new_observation, reward, is_done, truncated, _ = self.env.step(action)
            is_done = is_done or truncated
            if display:
                cv2.imshow('screen', cv2.resize(cv2.cvtColor(new_observation, cv2.COLOR_BGR2RGB), (480, 630)))
                cv2.waitKey(30)
            for i in range(self.skip_k_frame - 1):
                if not is_done:
                    self.obs_raw, reward_k, is_done, truncated, _ = self.env.step(action)
                    is_done = is_done or truncated
                    reward += reward_k
                    if display:
                        cv2.imshow('screen', cv2.resize(cv2.cvtColor(self.obs_raw, cv2.COLOR_BGR2RGB), (480, 630)))
                        cv2.waitKey(30)
                else:
                    self.obs_raw = self.env.reset()
                    self.agent.phi_reset()
                    self.last_episodic_reward = self.reward_recorder
                    self.last_episodic_steps = self.steps_recorder
                    self.reward_recorder = 0
                    self.steps_recorder = 0
                    self.episode_num += 1
                    if display:
                        cv2.imshow('screen', cv2.resize(cv2.cvtColor(self.obs_raw, cv2.COLOR_BGR2RGB), (480, 630)))
                        cv2.waitKey(30)
                    break
            next_phi = self.agent.phi_concat(self.obs_raw)
            self.agent.receive([self.phi_np, action, reward, is_done, next_phi])
            self.reward_recorder += reward
            self.steps_recorder += 1
            self.phi_np = next_phi
            steps_num += 1
        # return data_list

    def play_rounds(self, max_steps_num: int, display=False) -> []:
        if self.workers_num <= 1:
            return self.__play_serially(max_steps_num, display)


if __name__ == '__main__':
    # env = gym.make('Breakout-v0')
    # agent = AgentDQN(env, algorithm_version='2015')
    # agent.learning(epsilon_max=1.0, epsilon_min=0.01)
    now = int(round(time.time() * 1000))
    now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
    env_name = "ALE/Boxing-v5"
    env_ = DQNGym(env_name)
    exp_name = now02 + "_" + env_name.replace('/', '_')
    agent = AgentDQN(env_.action_dim, os.path.join('./model/', exp_name))
    dqn_play_ground = DQNPlayGround(env_, agent)
    frame_num = 0
    last_record_episode = None
    frame_num_last_record = 0
    while dqn_play_ground.episode_num <= args.episodes_num:
        dqn_play_ground.play_rounds(1)
        frame_num += 1
        agent.learn()
        if len(agent.memory) > args.init_data_size and agent.update_steps % 500000 == 9:
            print('episode ' + str(dqn_play_ground.episode_num) + ' total reward: ' +
                  str(dqn_play_ground.last_episodic_reward))
            agent.record_reward(dqn_play_ground.last_episodic_steps, dqn_play_ground.last_episodic_reward,
                                agent.epsilon, dqn_play_ground.episode_num - 9)
            agent.save()
            frame_num_last_record = frame_num
