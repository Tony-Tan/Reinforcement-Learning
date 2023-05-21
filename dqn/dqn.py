import gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from core.nn_utils import *
from core.agent import *
from core.training import *
from core.utils import *

args_list = [
    ['--env_name', 'InvertedDoublePendulum-v2', str, 'atari game name'],
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
    ['--epsilon_min', 0.1, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--epsilon_for_test', 0.05, float, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],
    ['--model_saving_period', 80000, int, 'Mujoco Gym environment，default: InvertedDoublePendulum-v2'],

]
args = script_args(args_list, 'dqn training arguments')


class DQNGym(Env):
    def __init__(self, env_name):
        super().__init__()
        self.env = gym.make(env_name)
        self.action_dim = self.env.action_space.n

    def step(self, action: np.ndarray) -> tuple:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs


class DQNCritic(nn.Module):
    def __init__(self, input_channel_size, output_size):
        super(DQNCritic, self).__init__()
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


def obs_pre_process(obs: np.ndarray):
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
    gray_img = gray_img / 128. - 1.
    return gray_img


class DQN(Agent):
    def __init__(self, action_dim: int, save_path: str):
        # basic elements settings
        super().__init__(args.memory_length)
        self.critic = DQNCritic(args.phi_temp_size, action_dim)
        self.action_n = action_dim
        self.epsilon = 1.0
        self.episodes_num = args.episodes_num
        self.phi = deque(maxlen=args.phi_temp_size)
        self.phi_np = None
        self.skip_k_frame = args.k_frames
        self.skip_k_frame_counter = 0
        self.skip_k_frame_reward_sum = 0
        self.last_action = 0
        self.update_steps = 1
        # self.last_10_episodic_reward = deque(maxlen=10)
        # self.last_10_episodic_steps = deque(maxlen=10)
        # self.last_episodic_reward = 0
        # self.last_episodic_steps = 0
        # critic networks settings
        self.action_dim = action_dim
        self.value_nn = DQNCritic(args.phi_temp_size, self.action_dim)
        self.target_value_nn = DQNCritic(args.phi_temp_size, self.action_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mini_batch_size = args.mini_batch_size
        self.optimizer = optim.Adam(self.value_nn.parameters(), lr=args.learning_rate)
        self.value_nn.to(self.device)
        self.target_value_nn.to(self.device)
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.recoder = Recorder(self.save_path)
        self.model_path = self.save_path
        self.phi_reset()

    def observe(self, obs, action, reward, terminated, truncated, inf, save_obs=True):
        if terminated or truncated:
            self.phi_reset()
            self.skip_k_frame_counter = 0
            self.skip_k_frame_reward_sum = 0
            # recorder
            # self.last_10_episodic_reward.append(self.last_episodic_reward)
            # self.last_10_episodic_steps.append(self.last_episodic_steps)
            # self.last_episodic_reward = 0
            # self.last_episodic_steps = 0

        elif (self.skip_k_frame_counter % self.skip_k_frame) == 0 and save_obs:
            self.skip_k_frame_reward_sum += reward
            self.memory.append([self.phi_np, action, self.skip_k_frame_reward_sum, terminated, truncated,
                                np.zeros_like(self.phi_np)])
            if len(self.memory) > 1:
                self.memory[-2][-1] = self.phi_np
            self.skip_k_frame_counter += 1
            self.skip_k_frame_reward_sum = 0
            # self.last_episodic_reward += reward
            # self.last_episodic_steps += 1
        else:
            self.skip_k_frame_counter += 1
            self.skip_k_frame_reward_sum += reward
            # self.last_episodic_reward += reward
            # self.last_episodic_steps += 1

    def react(self, obs: np.ndarray, testing=False):
        if (self.skip_k_frame_counter % self.skip_k_frame) == 0:
            obs = obs_pre_process(obs)
            # cv2.imshow('test', obs)
            # cv2.waitKey(10)
            self.phi.append(obs)
            self.phi_np = np.array(self.phi).astype(np.float32)
            with torch.no_grad():
                phi = torch.as_tensor(self.phi_np.astype(np.float32)).to(self.device)
                self.last_action = self.generate_action(phi, args.epsilon_for_test if testing else self.epsilon)
        return self.last_action

    def phi_reset(self):
        self.phi.clear()
        for i in range(args.phi_temp_size):
            self.phi.append(np.zeros([args.input_frame_width, args.input_frame_width]))

    def load(self, model_path, map_location=torch.device('cpu')):
        self.value_nn.load_state_dict(torch.load(model_path, map_location))

    def save(self):
        # save model files
        standard_info_print('model saved')
        now = int(round(time.time() * 1000))
        now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
        torch.save(self.value_nn.state_dict(), self.save_path + '/' + now02 + '_value.pth')
        torch.save(self.target_value_nn.state_dict(), self.save_path + '/' + now02 + '_value_target.pth')

    def generate_action(self, phi_t: torch.Tensor, epsilon: float = None):
        obs_input = phi_t.unsqueeze(0)
        state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
        value_of_action_list = state_action_values[0]
        return epsilon_greedy(value_of_action_list, epsilon)

    def synchronize_q_network(self):
        self.target_value_nn.load_state_dict(self.value_nn.state_dict())

    def learn(self):
        if len(self.memory) > args.init_data_size:
            samples = self.memory.sample(self.mini_batch_size)
            obs_array = []
            action_array = []
            next_obs_array = []
            reward_array = []
            is_done_array = []
            for sample_i in samples:
                # obs, action, self.skip_k_frame_reward_sum,
                # terminated, truncated, next_obs
                obs_array.append(sample_i[0])
                action_array.append([sample_i[1]])
                reward_array.append(sample_i[2])
                is_done_array.append(sample_i[3])
                next_obs_array.append(sample_i[5])
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
            max_next_state_value = np.array(max_next_state_value).astype(np.float32)
            max_next_state_value = (1.0 - is_done_array) * max_next_state_value
            reward_array = torch.from_numpy(reward_array)
            reward_array = torch.clamp(reward_array, min=-1., max=1.)
            q_value = reward_array + args.gamma * max_next_state_value

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
            if self.update_steps % args.model_saving_period == 0:
                self.save()
            self.epsilon = 1. - self.update_steps * 0.0000009
            self.epsilon = max(args.epsilon_min, self.epsilon)


def execute():
    now = int(round(time.time() * 1000))
    now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
    env_name = "ALE/Pong-v5"
    env_ = DQNGym(env_name)
    exp_name = now02 + "_" + env_name.replace('/', '_')
    agent_path = os.path.join('./exp/', exp_name)
    agent = DQN(env_.action_dim, agent_path)
    training = Training(env_, agent, max_episodes=args.episodes_num, test_period_steps=40000,
                        log_path=agent_path)
    training.online_training()


if __name__ == '__main__':
    execute()
