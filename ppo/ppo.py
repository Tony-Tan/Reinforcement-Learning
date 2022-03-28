import torch
import numpy as np
import environments.generate_trajectories_set as gts
import torch.nn as nn

TETRIS_WIDTH = 6
TETRIS_HEIGHT = 5

def features(state):
    state_bg = (state[0].reshape(1, -1).astype(np.float32) - 0.5) * 2
    state_t = np.array([state[1], state[2] / 360., state[3][0] / TETRIS_WIDTH,
                        state[3][1] / TETRIS_HEIGHT, 1.0]).astype(np.float32)
    state_ = np.append(state_bg, state_t)
    x_tensor_current = torch.from_numpy(state_)
    x_tensor_current.requires_grad = False
    return x_tensor_current


class policy_nn(nn.Module):
    def __init__(self, input_size):
        self.flatten = nn.Flatten()
        super(policy_nn, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.ReLU(),
            nn.Softmax())

    def forward(self,x):
        x = self.flatten(x)
        p = self.linear_relu_stack(x)
        return p





class PPO_Agent:
    def __init__(self, env_, trajectory_num_per_update_):
        self.env = env_
        self.trajectory_num_per_update = trajectory_num_per_update_
        pass

    def policy(self):
        pass

    def policy_num(self):
        pass

    def optimize(self, epoch, ):
        for epoch_i in range(1, epoch):
            # collect set of N trajectories
            trajectory_collection, reward_sum = gts.generate_trajectory_set(self.env, self.trajectory_num_per_update,
                                                                            self.policy_num, features)