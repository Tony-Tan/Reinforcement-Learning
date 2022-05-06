import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import gym
import matplotlib.pyplot as plt

# from gym.envs.mujoco import mujoco_env

TETRIS_WIDTH = 6
TETRIS_HEIGHT = 5
GAUSSIAN_NORM = 1. / (1. * np.sqrt(2 * np.pi))


def loss_function(pi_new, pi_old, estimate_advantage, epsilon):
    r_t = pi_new / pi_old
    l_clip = torch.min(r_t * estimate_advantage,
                       torch.clip(r_t, 1 - epsilon, 1 + epsilon) * estimate_advantage)
    return torch.mean(l_clip)


# def loss_function_1(action, pi_new_clip, pi_old_clip, estimate_advantage_clip,
#                   epsilon_clip, value_output, value_label, c_1, c_2):
#     r_t = pi_new_clip / pi_old_clip
#     l_clip = torch.min(r_t * estimate_advantage_clip,
#                        torch.clip(r_t, 1 - epsilon_clip, 1 + epsilon_clip) * estimate_advantage_clip)
#     l_vf = torch.pow(value_output - value_label, 2)
#     entropy = -torch.sum(pi_new_clip * torch.log(pi_new_clip))
#     return torch.mean(l_clip - c_1 * l_vf  + c_2 * entropy)


class policy_nn(nn.Module):
    def __init__(self, input_size):
        super(policy_nn, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh())

    def forward(self, x):
        x = self.flatten(x)
        mu = self.linear_mlp_stack(x)
        return mu


class value_nn(nn.Module):
    def __init__(self, input_size):
        super(value_nn, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        x = self.flatten(x)
        value = self.linear_stack(x)
        return value


class StepSet(Dataset):
    def __init__(self, trajectory_set, transform=None):
        self.trajectory_set = trajectory_set
        self.transform = transform
        self.step_set = []
        for trajectory_i in trajectory_set:
            for step_i in trajectory_i:
                self.step_set.append(step_i)

    def __len__(self):
        return len(self.step_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'state': self.step_set[idx][0],
                  'action': self.step_set[idx][1],
                  'reward': self.step_set[idx][2],
                  'next_state': self.step_set[idx][3],
                  'action_likelihood': self.step_set[idx][4],
                  'G': self.step_set[idx][5]
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample


def trajectory_return(trajectory_set, gamma=0.99):
    for trajectory_i in trajectory_set:
        n = len(trajectory_i)
        trajectory_i[n-1].append(trajectory_i[n-1][2])
        for step_i in reversed(range(n-1)):
            trajectory_i[step_i].append(trajectory_i[step_i][2]+gamma*trajectory_i[step_i+1][2])


class PPO_Agent:
    def __init__(self, env_, trajectory_set_size_=100):
        self.env = env_
        self.module_input_size = len(self.env.reset())
        self.trajectory_set_size = trajectory_set_size_
        self.policy_module = policy_nn(self.module_input_size)
        self.value_module = value_nn(self.module_input_size)
        pass

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
        # TODO generate mujoco collection
        trajectory_set = []
        for set_i in range(set_size):
            trajectory = []
            state = self.env.reset()
            for i in range(horizon):
                x_tensor = torch.tensor(state)
                action, action_likelihood = self.action_selection(x_tensor)
                new_state, reward, is_done, _ = self.env.step(action)
                trajectory.append([state, action, reward, new_state, action_likelihood])
                state = new_state
            trajectory_set.append(trajectory)
        return trajectory_set

    def optimize(self, policy_update_times, horizon):
        for update_i in range(1, policy_update_times):
            # collect set of N trajectories
            # trajectory_collection, reward_sum = gts.generate_trajectory_set(self.env, self.trajectory_set_size,
            #                                                                 features,
            #                                                                 select_action_fn=self.action_selection)
            trajectory_collection = self.generate_trajectory_set(set_size=10, horizon=1000)
            trajectory_return(trajectory_collection)
            steps_dataset = StepSet(trajectory_collection, transform=transforms.Compose([transforms.ToTensor()]))

            # update value
            dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=4, shuffle=True, num_workers=4)
            optimizer = torch.optim.SGD(self.value_module.parameters(), lr=0.001)
            loss = torch.nn.MSELoss()
            for batch_i, data_i in enumerate(dataset_loader):
                print(batch_i)
                current_state = data_i['state']
                value = data_i['G']
                estimate_value = self.value_module(current_state)
                residual = loss(estimate_value,value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # update policy
            dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=4, shuffle=True, num_workers=4)
            optimizer = torch.optim.SGD(self.policy_module.parameters(), lr=0.001)
            for batch_i, data_i in enumerate(dataset_loader):
                print(batch_i)
                current_state = data_i['state']
                action = data_i['action']
                reward = data_i['reward']
                next_state = data_i['next_state']
                action_lh = data_i['action_likelihood']
                mu = self.policy_module(current_state)
                action_likelihood = GAUSSIAN_NORM * torch.exp(torch.pow(action-mu, 2))
                # calculate advantage
                # TODO
                advantage = reward + self.value_module(next_state) - self.value_module(current_state)
                loss = - loss_function(action_likelihood, action_lh, advantage, 0.2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    env = gym.make('Ant-v2')
    # pp0_agent = PPO_Agent(env, 10)
    state_ = env.reset()
    print(len(state_))
    for _ in range(100):
        action_ = 0
        new_state, reward_, is_done, _ = env.step(action_)
    env.close()
    pass
