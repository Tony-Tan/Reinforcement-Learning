import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import gym
import os
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity


def print_time():
    print('Time step:' + time.strftime("%H:%M:%S", time.localtime()))


def loss_function(pi_new, pi_old, estimate_advantage, epsilon):
    r_t = pi_new / pi_old
    r_advantage = r_t * estimate_advantage
    clip_advantage = torch.clip(r_t, 1. - epsilon, 1. + epsilon) * estimate_advantage
    output_clip = torch.min(r_advantage, clip_advantage)
    return - torch.mean(output_clip, dim=0)


class PolicyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.flatten(x)
        mu = self.linear_mlp_stack(x)
        return mu


class ValueNN(nn.Module):
    def __init__(self, input_size):
        super(ValueNN, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x = self.flatten(x)
        value = self.linear_stack(x)
        return value


class ValueDataset(Dataset):
    def __init__(self, data_buffer, transform=None):
        self.data_buffer = data_buffer
        self.len = data_buffer.buffer_size
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'state': self.data_buffer.dataset['state'][idx],
                  'G': self.data_buffer.dataset['G'][idx]
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample


class PolicyDataset(Dataset):
    def __init__(self, data_buffer, with_value=False, transform=None):
        self.data_buffer = data_buffer
        self.with_value = with_value
        self.transform = transform
        self.len = data_buffer.buffer_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'state': self.data_buffer.dataset['state'][idx],
                  'action': self.data_buffer.dataset['action'][idx],
                  'action_likelihood': self.data_buffer.dataset['action_likelihood'][idx],
                  'GAE': self.data_buffer.dataset['GAE'][idx]
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensorValue(object):
    def __call__(self, sample):
        state = sample['state']
        g = sample['G']
        return {'state': torch.from_numpy(state),
                'G': torch.from_numpy(g)
                }


class ToTensorPolicy(object):
    def __call__(self, sample):
        state = sample['state']
        action = sample['action']
        action_lh = sample['action_likelihood']
        gae = sample['GAE']
        return {'state': torch.from_numpy(state),
                'action': torch.from_numpy(action),
                'action_likelihood': torch.from_numpy(action_lh),
                'GAE': torch.from_numpy(gae)
                }


def cal_return(reward_np, termination_list, gamma=0.99):
    step_size = len(termination_list)
    G = np.zeros(step_size, dtype=np.float32)
    G[-1] = reward_np[-1]
    for step_i in reversed(range(step_size - 1)):
        reward_i = reward_np[step_i]
        if termination_list[step_i] is True:
            G[step_i] = reward_i
        else:
            G[step_i] = reward_i + gamma * G[step_i + 1]
    return G.reshape((-1, 1))


def g_a_e(termination_list, reward_list, state_value_np, lambda_value=0.95, gamma=0.99):
    term_np = np.array(termination_list, dtype=np.float32).reshape(-1, 1)
    reward_np = np.array(reward_list, dtype=np.float32).reshape(-1, 1)
    delta_np = reward_np - state_value_np
    delta_np[:-1] = delta_np[:-1] + gamma * state_value_np[1:] * (1. - term_np[:-1])
    # generate advantage
    n = len(termination_list)
    advantage = np.zeros(n, dtype=np.float32)
    advantage[n - 1] = delta_np[n - 1]
    for i in reversed(range(n - 1)):
        if termination_list[i] is True:
            advantage[i] = delta_np[i]
        else:
            advantage[i] = delta_np[i] + gamma * lambda_value * advantage[i + 1]

    std = np.std(advantage)
    mean = np.mean(advantage)
    advantage = (advantage - mean) / std
    return advantage


class StepBuffer:
    def __init__(self):
        self.buffer_size = 0
        self.buffer = {'state': [],
                       'action': [],
                       'reward': [],
                       'action_likelihood': [],
                       'termination': [],
                       'G': [],
                       'state_value': [],
                       'GAE': []}
        self.dataset = {}

    def store_experience(self, state, action, reward, action_lh, termination):
        self.buffer_size += 1
        self.buffer['state'].append(state)
        self.buffer['action'].append(action)
        self.buffer['reward'].append(reward)
        self.buffer['action_likelihood'].append(action_lh)
        self.buffer['termination'].append(termination)

    def store_state_value(self, state_value_np):
        self.dataset['state_value'] = state_value_np.reshape((-1, 1))

    def generate_value_dataset(self, reward_std=1):
        self.dataset['state'] = np.array(self.buffer['state'], dtype=np.float32)
        # reward normalization according to
        # Burda et al., “Large-Scale Study of Curiosity-Driven Learning.”
        self.dataset['reward'] = np.array(self.buffer['reward'], dtype=np.float32)/reward_std
        g_np = cal_return(self.dataset['reward'], self.buffer['termination'])
        self.dataset['G'] = g_np

    def generate_policy_dataset(self):
        self.dataset['GAE'] = g_a_e(self.buffer['termination'], self.buffer['reward'],
                                    self.dataset['state_value']).reshape((-1, 1))
        self.dataset['action'] = np.array(self.buffer['action'], dtype=np.float32).reshape((-1, 1))
        self.dataset['action_likelihood'] = np.array(self.buffer['action_likelihood'],
                                                     dtype=np.float32).reshape((-1, 1))


class PPOAgent:
    def __init__(self, env_name, gamma=0.99, model_path='./data/models/'):
        self.env_name = env_name
        self.gamma = gamma
        env_ = gym.make(self.env_name)
        self.module_input_size = len(env_.reset())
        self.action_size = env_.action_space.shape[0]
        self.state_mean = None
        self.state_std = None
        # self.reward_mean = 0
        self.reward_std = 1
        self.action_std = 1.
        self.gaussian_normalize = 1. / (self.action_std * np.sqrt(2 * np.pi))
        self.log_writer = None
        self.policy_module = None
        self.value_module = None
        self.start_update_i = 1
        model_name_file_path = os.path.join(model_path, 'last_models.txt')
        if os.path.exists(model_name_file_path):
            model_rec_file = open(model_name_file_path, 'r')
            model_name = model_rec_file.readline().strip('\n')
            self.start_update_i = int(model_rec_file.readline().strip('\n')) + 1
            model_rec_file.close()
            self.load_model(model_name, model_path)
        else:
            self.policy_module = PolicyNN(self.module_input_size, self.action_size)
            self.value_module = ValueNN(self.module_input_size)
            self.env_info(model_path)

    def env_info(self, data_path):
        env = gym.make(self.env_name)
        state_list = []
        discount_return_list = []

        for i in range(1000):
            state = env.reset()
            state_list.append(state)
            total_reward = 0
            step_i = 0
            while True:
                random_action = env.action_space.sample()
                new_state, reward, is_done, _ = env.step(random_action)
                state_list.append(new_state)
                total_reward += np.power(self.gamma, step_i)*reward
                step_i += 1
                if is_done:
                    break
            discount_return_list.append(total_reward)
        env.close()
        reward_np = np.array(discount_return_list, dtype=np.float32)
        state_np = np.array(state_list, dtype=np.float32)
        self.reward_std = np.std(reward_np) + 1e-5  # not be zero
        self.state_std = np.std(state_np, axis=0) + 1e-5  # not be zero
        self.state_mean = np.mean(state_np, axis=0)
        state_mean_path = os.path.join(data_path, 'state_mean.npy')
        np.save(state_mean_path, self.state_mean)
        state_std_path = os.path.join(data_path, 'state_std.npy')
        np.save(state_std_path, self.state_std)
        reward_std_path = os.path.join(data_path, 'reward_std.npy')
        np.save(reward_std_path, self.reward_std)
        print('state reward info saved!')

    def policy(self, x_tensor):
        with torch.no_grad():
            mu = self.policy_module(x_tensor)
            return mu

    def update_action_std(self):
        self.action_std = max(self.action_std * 0.993, 0.1)
        self.gaussian_normalize = 1. / (self.action_std * np.sqrt(2 * np.pi))

    def action_selection(self, x_tensor):
        with torch.no_grad():
            self.policy_module.eval()
            mu = self.policy(x_tensor)
            mu_np = mu.clone().detach().cpu().numpy()
            action = np.random.normal(mu_np, self.action_std)
            pro = self.gaussian_normalize * np.exp(-0.5 * ((action - mu_np) / self.action_std) ** 2)
            return action, pro

    def generate_trajectory_set(self, actor_num, horizon):
        data_buffer = StepBuffer()
        with torch.no_grad():
            self.policy_module.to('cpu')
            env_ = gym.make(self.env_name)
            state = env_.reset()
            while data_buffer.buffer_size < horizon:
                state_normalized = (state-self.state_mean)/self.state_std
                x_tensor = torch.tensor(state_normalized, dtype=torch.float32).to('cpu')
                action, action_likelihood = self.action_selection(x_tensor)
                new_state, reward, is_done, _ = env_.step(action)
                data_buffer.store_experience(state_normalized, action, reward, action_likelihood, is_done)
                if is_done:
                    new_state = env_.reset()
                state = new_state
            return data_buffer

    def load_model(self, model_name, path='./data/models/'):
        policy_module_path = os.path.join(path, model_name) + '_policy.pt'
        self.policy_module = torch.load(policy_module_path)
        value_module_path = os.path.join(path, model_name) + '_value.pt'
        self.value_module = torch.load(value_module_path)
        state_mean_path = os.path.join(path, 'state_mean.npy')
        self.state_mean = np.load(state_mean_path)
        state_std_path = os.path.join(path, 'state_std.npy')
        self.state_std = np.load(state_std_path)
        reward_std_path = os.path.join(path, 'reward_std.npy')
        self.reward_std = np.load(reward_std_path)
        print('model loaded: ' + model_name)
        print('state mean', self.state_mean)
        print('state std', self.state_std)
        print('reward std', self.reward_std)

    def save_model(self, update_times, path='./data/models/'):
        model_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        policy_module_path = os.path.join(path, model_name) + '_policy.pt'
        torch.save(self.policy_module, policy_module_path)
        value_module_path = os.path.join(path, model_name) + '_value.pt'
        torch.save(self.value_module, value_module_path)
        model_rec_file = open(os.path.join(path, 'last_models.txt'), 'w+')
        model_rec_file.write(model_name + '\n')
        model_rec_file.write(str(update_times) + '\n')
        model_rec_file.close()

        print('model saved: ' + model_name)

    def optimize_value_update(self, update_i, data_buffer, value_batch_size=64, value_regression_epoch=15,
                              value_lr=1e-3, device_='cuda'):
        data_buffer.generate_value_dataset(self.reward_std)
        steps_dataset = ValueDataset(data_buffer, transform=ToTensorValue())

        dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=value_batch_size,
                                                     shuffle=True, num_workers=0, drop_last=True)
        # update value
        self.value_module.to(device_)
        optimizer_value = torch.optim.SGD(self.value_module.parameters(), lr=value_lr)
        loss = torch.nn.MSELoss()
        average_residual = 0
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for i in range(value_regression_epoch):
            for batch_i, data_i in enumerate(dataset_loader):
                current_state = data_i['state'].to(device_)
                value = data_i['G'].to(device_)
                estimate_value = self.value_module(current_state)
                residual = loss(estimate_value, value)
                optimizer_value.zero_grad()
                residual.backward()
                optimizer_value.step()
                average_residual += residual.item()
        #
        if update_i % 200 == 0:
            average_residual /= (value_regression_epoch * int(len(steps_dataset) / value_batch_size))
            print_time()
            print('\t\t regression state value for advantage; epoch: ' + str(update_i))
            print('\t\t value loss: ' + str(average_residual))
            print('-----------------------------------------------------------------')
            self.log_writer.add_scalar('value loss', average_residual, update_i)

        value_data_array = None
        dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=40000,
                                                     shuffle=False, num_workers=0, drop_last=False)
        with torch.no_grad():
            for batch_i, data_i in enumerate(dataset_loader):
                current_state = data_i['state'].to(device_)
                state_value = self.value_module(current_state)
                if value_data_array is None:
                    value_data_array = state_value.cpu().numpy()
                else:
                    value_data_array = np.vstack((value_data_array, state_value.cpu().numpy()))
        data_buffer.store_state_value(value_data_array)

    def optimize_policy_update(self, data_buffer, policy_update_epoch=10, policy_lr=1e-6, epsilon=0.2, device_='cuda'):
        data_buffer.generate_policy_dataset()
        steps_dataset = PolicyDataset(data_buffer, transform=ToTensorPolicy())

        self.policy_module.train()
        self.policy_module.to(device_)
        dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=64, shuffle=True,
                                                     num_workers=0, drop_last=True)
        optimizer_policy = torch.optim.SGD(self.policy_module.parameters(), lr=policy_lr)
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for mini_epoch_i in range(policy_update_epoch):
            for batch_i, data_i in enumerate(dataset_loader):
                current_state = data_i['state'].to(device_)
                action = data_i['action'].to(device_)
                action_lh_old = data_i['action_likelihood'].to(device_)
                mu = self.policy_module(current_state)
                action_std_vers = 1. / self.action_std
                action_lh_new = self.gaussian_normalize * torch.exp(
                    torch.mul(-0.5, torch.pow((action - mu) * action_std_vers, 2)))
                advantage = data_i['GAE'].to(device_)
                loss = loss_function(action_lh_new, action_lh_old, advantage, epsilon)
                optimizer_policy.zero_grad()
                loss.backward(torch.ones_like(loss))
                optimizer_policy.step()
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    def optimize(self, policy_update_times, horizon=4000, actor_num=1):
        self.log_writer = SummaryWriter('./data/log/')
        print_time()
        print('\t\t optimizing started:')
        value_lr = 1e-3
        policy_lr = 1e-4
        for update_i in range(self.start_update_i, policy_update_times):
            data_buffer = self.generate_trajectory_set(actor_num, horizon)
            # == log ==
            # ------------------------
            self.optimize_value_update(update_i, data_buffer, value_lr=value_lr)
            # generated advantage estimation
            # update policy
            self.optimize_policy_update(data_buffer, policy_lr=policy_lr)
            # print log
            if update_i % 1000 == 0:
                test_cumulate_reward = self.log_record(update_i)
                self.save_model(update_i)
                self.update_action_std()
                value_lr = max(value_lr*0.96, 1e-6)
                policy_lr = max(policy_lr*0.96, 4e-6)
                self.log_writer.add_scalar('Variable STD', self.action_std, update_i)
                self.log_writer.add_scalar('value lr', value_lr, update_i)
                self.log_writer.add_scalar('policy lr', policy_lr, update_i)

    def log_record(self, update_i, device_='cpu'):
        env = gym.make(self.env_name)
        total_reward = 0.0
        total_steps = 0
        self.policy_module.eval()
        self.policy_module.to(device_)
        for i in range(1000):
            obs = env.reset()
            while True:
                x_tensor = torch.tensor((obs-self.state_mean)/self.state_std, dtype=torch.float32).to(device_)
                mu = self.policy(x_tensor)
                action = mu.clone().detach().cpu().numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                total_steps += 1
                if done:
                    break
        print("Episode done in %d steps, total reward %.2f" % (
            total_steps / 1000, total_reward / 1000))
        env.close()
        self.log_writer.add_scalar('reward', total_reward / 1000., update_i)
        self.log_writer.add_scalar('step', total_steps / 1000., update_i)
        return total_reward / 1000.

# Hyperparameter          Value
# Horizon (T)             2048
# Adam step size          3e−4
# Num. epochs               10
# Minibatch size             64
# Discount ( γ )           0.99
# GAE parameter (λ)         0.95


if __name__ == '__main__':
    ppo = PPOAgent('Swimmer-v4')
    ppo.optimize(policy_update_times=1000000)
    print_time()
