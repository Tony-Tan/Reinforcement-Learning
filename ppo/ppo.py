import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import gym
import os
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./data/log/')
GAUSSIAN_NORM = 1. / (1. * np.sqrt(2 * np.pi))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def print_time():
    print('Time step:'+time.strftime("%H:%M:%S", time.localtime()))


def loss_function(pi_new, pi_old, estimate_advantage, epsilon):
    r_t = pi_new / pi_old
    l_clip = torch.min(r_t * estimate_advantage,
                       torch.clip(r_t, 1 - epsilon, 1 + epsilon) * estimate_advantage)
    return - torch.mean(l_clip, dim=0)


class PolicyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        mu = self.linear_mlp_stack(x)
        return mu


class ValueNN(nn.Module):
    def __init__(self, input_size):
        super(ValueNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        value = self.linear_stack(x)
        return value


class StepSet(Dataset):
    def __init__(self, trajectory_set, with_value=False):
        self.trajectory_set = trajectory_set
        self.with_value = with_value
        # self.transform = transform
        self.step_set = []
        for trajectory_i in trajectory_set:
            for step_i in trajectory_i:
                self.step_set.append(step_i)

    def __len__(self):
        return len(self.step_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not self.with_value:
            sample = {'state': self.step_set[idx][0],
                      'action': self.step_set[idx][1],
                      'reward': self.step_set[idx][2],
                      'next_state': self.step_set[idx][3],
                      'action_likelihood': self.step_set[idx][4],
                      'G': self.step_set[idx][5]
                      }
            return sample
        else:
            sample = {'state': self.step_set[idx][0],
                      'action': self.step_set[idx][1],
                      'reward': self.step_set[idx][2],
                      'next_state': self.step_set[idx][3],
                      'action_likelihood': self.step_set[idx][4],
                      'G': self.step_set[idx][5],
                      'state_value': self.step_set[idx][6],
                      'next_state_value': self.step_set[idx][7]
                      }
            return sample


def trajectory_return(trajectory_set, gamma=0.99):
    for trajectory_i in trajectory_set:
        n = len(trajectory_i)
        for step_i in reversed(range(n-1)):
            g = np.array([trajectory_i[step_i][2]+gamma*trajectory_i[step_i+1][2]]).astype(np.float32)
            trajectory_i[step_i].append(g)
        g = np.array([trajectory_i[n - 1][2]]).astype(np.float32)
        trajectory_i[n - 1].append(g)


class PPOAgent:
    def __init__(self, env_name, model_name=None, model_path='./data/models/'):
        self.env_name = env_name
        env_ = gym.make(self.env_name)
        self.module_input_size = len(env_.reset())
        self.action_size = env_.action_space.shape[0]
        self.policy_module = None
        self.value_module = None
        if model_name is not None:
            self.load_model(model_name, model_path)
        else:
            self.policy_module = PolicyNN(self.module_input_size, self.action_size)
            self.value_module = ValueNN(self.module_input_size)
        self.policy_module.to(device)
        self.value_module.to(device)

    def policy(self, x_tensor):
        with torch.no_grad():
            mu = self.policy_module(x_tensor)
            return mu

    def action_selection(self, x_tensor):
        with torch.no_grad():
            mu = self.policy(x_tensor)
            mu_np = mu.clone().detach().cpu().numpy()
            action = np.random.normal(mu_np, 1)
            pro = GAUSSIAN_NORM*np.exp(-0.5*(action-mu_np)**2)
            return action, pro

    def generate_trajectory_set(self, set_size, horizon):
        with torch.no_grad():
            print('start generating trajectory')
            print_time()
            trajectory_set = []
            env_list = []
            total_reward = 0

            for set_i in range(set_size):
                env_ = gym.make(self.env_name)
                env_list.append(env_)
                trajectory_set.append([])
            state_list = []
            for env_i in env_list:
                state = env_i.reset()
                state_np = np.array([state]).astype(np.float32)
                state_list.append(state_np)

            for i in range(horizon):
                state_list_np = np.array(state_list).astype(np.float32)
                # x_tensor = torch.tensor(state_list_np, dtype=torch.float32).to(device_)
                x_tensor = torch.tensor(state_list_np, dtype=torch.float32).to(device)
                action, action_likelihood = self.action_selection(x_tensor)
                state_list = []
                for env_i in range(len(env_list)):
                    new_state, reward, is_done, _ = env_list[env_i].step(action[env_i])
                    action_np = np.float32(action[env_i])
                    reward_np = np.float32(reward)
                    action_likelihood_np = np.float32(action_likelihood[env_i])
                    new_state_np = np.array([new_state]).astype(np.float32)
                    trajectory_set[env_i].append([state_np, action_np, reward_np,
                                                  new_state_np, action_likelihood_np])
                    state_list.append(new_state_np)
                    total_reward += reward_np

            print('end generating trajectory')
            print_time()

            return trajectory_set, total_reward

    def load_model(self, model_name, path='./data/models/'):
        policy_module_path = os.path.join(path, model_name) + '_policy.pt'
        self.policy_module = torch.load(policy_module_path)
        value_module_path = os.path.join(path, model_name) + '_value.pt'
        self.value_module = torch.load(value_module_path)
        print('model loaded: ' + model_name)

    def save_model(self, path='./data/models/'):
        model_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        policy_module_path = os.path.join(path, model_name) + '_policy.pt'
        torch.save(self.policy_module, policy_module_path)
        value_module_path = os.path.join(path, model_name) + '_value.pt'
        torch.save(self.value_module, value_module_path)
        print('model saved: ' + model_name)

    def optimize(self, policy_update_times, horizon=2048, actor_num=8):
        log_reward = 0
        for update_i in range(1, policy_update_times):
            trajectory_collection, total_reward = self.generate_trajectory_set(actor_num, horizon)
            # == log ==
            log_reward += total_reward
            # ------------------------
            trajectory_return(trajectory_collection)
            steps_dataset = StepSet(trajectory_collection)
            dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=horizon,
                                                         shuffle=True, num_workers=0)
            # update value
            optimizer_value = torch.optim.SGD(self.value_module.parameters(), lr=0.001)
            loss = torch.nn.MSELoss()
            residual_log = 0
            for mini_epoch_i in range(max(50-update_i, 1)):
                residual = None
                for batch_i, data_i in enumerate(dataset_loader):
                    current_state = data_i['state'].to(device)
                    value = data_i['G'].to(device)
                    estimate_value = self.value_module(current_state)
                    residual = loss(estimate_value, value)
                    optimizer_value.zero_grad()
                    residual.backward()
                    optimizer_value.step()
                residual_log = residual.item()
            if update_i % 10 == 0:
                print('-----------------------------------------------------------------')
                print('regression state value for advantage; epoch: ' + str(update_i))
                print_time()
                print('value loss: ' + str(residual_log))
                writer.add_scalar('value loss', residual_log, update_i)

            # add value data into dataset
            value_data_array = []
            dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=horizon*actor_num,
                                                         shuffle=False, num_workers=0)
            with torch.no_grad():
                for batch_i, data_i in enumerate(dataset_loader):
                    current_state = data_i['state'].to(device)
                    next_state = data_i['next_state'].to(device)
                    state_value = self.value_module(current_state)
                    next_state_value = self.value_module(next_state)
                    value_data_array.append([state_value.numpy(), next_state_value.numpy()])
            total_step_index = 0
            for t_i in trajectory_collection:
                for step_i in t_i:
                    step_i.append(value_data_array[0][0][total_step_index])
                    step_i.append(value_data_array[0][1][total_step_index])
                    total_step_index += 1
            steps_dataset = StepSet(trajectory_collection, with_value=True)

            # update policy
            dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=64, shuffle=True, num_workers=0)
            optimizer_policy = torch.optim.Adam(self.policy_module.parameters(), lr=3e-4)
            for mini_epoch_i in range(10):
                for batch_i, data_i in enumerate(dataset_loader):
                    # print(batch_i)
                    current_state = data_i['state'].to(device)
                    action = data_i['action'].to(device)
                    reward = data_i['reward'].to(device)
                    # next_state = data_i['next_state'].to(device)
                    action_lh = data_i['action_likelihood'].to(device)
                    state_value = data_i['state_value'].to(device)
                    next_state_value = data_i['next_state_value'].to(device)
                    mu = self.policy_module(current_state)
                    action_likelihood = GAUSSIAN_NORM * torch.exp(- torch.mul(0.5, torch.pow(action-mu, 2)))
                    reward = reward.reshape(-1, 1)
                    advantage = reward + next_state_value - state_value
                    loss = loss_function(action_likelihood, action_lh, advantage, 0.2)
                    optimizer_policy.zero_grad()
                    loss.backward(torch.ones_like(loss))
                    optimizer_policy.step()

            # print log
            if update_i % 100 == 0:
                print('policy update: ' + str(update_i)+' time(s); reward: ' +
                      str(log_reward/(horizon * actor_num * 100)))
                writer.add_scalar('reward', log_reward/(horizon * actor_num * 100), update_i)
                self.save_model()
                log_reward = 0
                print('-------------------------------------------------------\n\n')


# Hyperparameter          Value
# Horizon (T)             2048
# Adam step size          3e−4
# Num. epochs               10
# Minibatch size             64
# Discount ( γ )           0.99
# GAE parameter (λ)         0.95

if __name__ == '__main__':
    ppo = PPOAgent('Swimmer-v2')
    ppo.optimize(policy_update_times=1000000)

