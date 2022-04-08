import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import gym
# from gym.envs.mujoco import mujoco_env

TETRIS_WIDTH = 6
TETRIS_HEIGHT = 5
GAUSSIAN_NORM = 1. / (1. * np.sqrt(2 * np.pi))


def loss_function(pi_new_clip, pi_old_clip, estimate_advantage_clip, epsilon_clip):
    r_t = pi_new_clip / pi_old_clip
    l_clip = torch.min(r_t * estimate_advantage_clip,
                       torch.clip(r_t, 1 - epsilon_clip, 1 + epsilon_clip) * estimate_advantage_clip)
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
        self.flatten = nn.Flatten()
        super(policy_nn, self).__init__()
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh())

    def forward(self, x):
        x = self.flatten(x)
        mu = self.linear_mlp_stack(x)
        return mu


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
                  'action_lh': self.step_set[idx][4]
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample


class PPO_Agent:
    def __init__(self, env_, trajectory_set_size_=100):
        self.env = env_
        self.trajectory_set_size = trajectory_set_size_
        self.policy_module = policy_nn(input_size=10)
        pass

    def policy(self, x_tensor):
        mu = self.policy_module(x_tensor)
        return mu

    def action_selection(self, x_tensor):
        mu = self.policy(x_tensor)
        mu_np = mu.clone().detach().cpu().numpy()
        action = np.random.normal(mu_np, 1)
        pro = GAUSSIAN_NORM*np.exp(-0.5*(action-mu_np)**2)
        return action, pro

    def generate_trajectory_set(self):
        # TODO generate mujoco collection
        return []
        pass

    def optimize(self, policy_update_times, ):
        for update_i in range(1, policy_update_times):
            # collect set of N trajectories
            # trajectory_collection, reward_sum = gts.generate_trajectory_set(self.env, self.trajectory_set_size,
            #                                                                 features,
            #                                                                 select_action_fn=self.action_selection)
            trajectory_collection = self.generate_trajectory_set()
            steps_dataset = StepSet(trajectory_collection, transform=transforms.Compose([transforms.ToTensor()]))
            dataset_loader = torch.utils.data.DataLoader(steps_dataset, batch_size=4, shuffle=True, num_workers=4)
            optimizer = torch.optim.SGD(self.policy_module.parameters(), lr=0.001)
            for batch_i, data_i in enumerate(dataset_loader):
                print(batch_i)
                current_state = data_i['state']
                action = data_i['action']
                reward = data_i['reward']
                next_state = data_i['next_state']
                action_lh = data_i['action_lh']
                mu = self.policy_module(current_state)
                policy_dist = GAUSSIAN_NORM * torch.exp(torch.pow(action-mu,2))
                loss = - loss_function(policy_dist,action_lh,reward,0.2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    # pp0_agent = PPO_Agent(env, 10)
    env.reset()
    for _ in range(100):
        env.render(mode='rgb_array')
        env.step(env.action_space.sample())
    env.close()
    pass
