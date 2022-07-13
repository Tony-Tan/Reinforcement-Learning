import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import time
import os
from core.exceptions import *
from core.basic import *
from torch.utils.tensorboard import SummaryWriter
GAUSSIAN_NORMAL_CONSTANT = np.float32(1./np.sqrt(2*np.pi))
LOG_GAUSSIAN_NORMAL_CONSTANT = np.log(GAUSSIAN_NORMAL_CONSTANT)


class Actor(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super(Actor, self).__init__()
        self.obs_dim = observation_space.shape[-1]
        self.action_dim = action_space.shape[-1]
        self.action_low = torch.tensor(np.array([i for i in action_space.low]), dtype=torch.float32)
        self.action_high = torch.tensor(np.array([i for i in action_space.high]), dtype=torch.float32)
        self.action_mean = torch.as_tensor((self.action_low + self.action_high) / 2.0, dtype=torch.float32)
        self.action_radius = torch.as_tensor((self.action_high - self.action_low) / 2.0, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        raise NotImplement

    def act(self, x: torch.Tensor, stochastically=True, with_log_pro=False):
        raise NotImplement


class Critic(nn.Module):
    def __init__(self, observation_space: gym.Space,
                 action_space: gym.Space):
        super(Critic, self).__init__()
        self.obs_dim = observation_space.shape[-1]
        self.action_dim = action_space.shape[-1]

    def forward(self, *args):
        raise NotImplement


class Env:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplement

    def step(self):
        raise NotImplement


class EnvModel:
    def __init__(self):
        pass


class Agent:
    def __init__(self, name: str, path='./data/models/'):

        self.agent_name = name
        self.actor = None
        self.critic = None
        self.start_epoch = 0
        self.path = path
        # self.checkpoint={
        #     'epoch': EPOCH,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': LOSS,
        #     }
        self.checkpoint = {}

    def reaction(self, obs: np.ndarray):
        raise NotImplement

    def simulation(self):
        raise NotImplement

    def save(self, epoch: int):
        raise NotImplement

    def load(self):
        """checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # - or -
        model.train()
        """
        raise NotImplement

    def update_actor(self, epoch_num: int, data: DataBuffer, device: str, log_writer: SummaryWriter):
        raise NotImplement

    def update_critic(self, epoch_num: int, data: DataBuffer, device: str, log_writer: SummaryWriter):
        raise NotImplement

    def update_actor_critic(self, *args):
        raise NotImplement


class RLExperiment:
    def __init__(self, env, gamma: float, agent: Agent, buffer_size: int,
                 data_template: dict, log_path: str):
        """
        :param env:
        :param agent:
        :param buffer_size:
        :param data_template: the data buffer template
        """
        self.env = env
        self.gamma = gamma
        self.agent = agent
        self.buffer = None

        self.buffer = DataBuffer(buffer_size, data_template)
        if log_path is not None:
            self.exp_log_writer = SummaryWriter(log_path)

    def generate_trajectories(self, total_step_num: int):
        if total_step_num > self.buffer.max_size:
            raise UnnecessaryCalculation
        if total_step_num <= 0:
            # step_num = self.buffer.max_size
            self.buffer.clean()
        step_num = 0
        current_obs = self.env.reset()
        while step_num < total_step_num:
            action = self.agent.actor.act(current_obs)
            new_obs, reward, is_done, _ = self.env.step(action)
            self.buffer.push([current_obs, action, reward, is_done])
            step_num += 1
            if is_done:
                current_obs = self.env.reset()
            else:
                current_obs = new_obs

    def play(self, *args):
        raise NotImplement

    def test(self, round_num: int, test_round_num: int, device: str):
        env = self.env
        total_reward = 0.0
        total_steps = 0
        self.agent.actor.eval()
        self.agent.actor.to(device)
        for i in range(test_round_num):
            obs = env.reset()
            while True:
                mu = self.agent.reaction(obs)
                action = mu.clone().detach().cpu().numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                total_steps += 1
                if done:
                    break
        print("Episode done in %d steps, total reward %.2f" % (
            total_steps / test_round_num, total_reward / test_round_num))
        env.close()
        self.exp_log_writer.add_scalar('reward', total_reward / test_round_num, round_num)
        self.exp_log_writer.add_scalar('step', total_steps / test_round_num, round_num)
        return total_reward / test_round_num


class MLPGaussianActor(Actor):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 hidden_layers_size: list, hidden_action_fc):
        """
        :param obs_dim:
        :param action_dim:
        :param hidden_layers_size:
        :param hidden_action_fc:
        :param output_action_fc: dict, {'mu':action, 'std':action}
        """
        super(MLPGaussianActor, self).__init__(observation_space, action_space)
        layers = [self.obs_dim, *hidden_layers_size]
        self.linear_mlp_stack = MLP(layers, hidden_action_fc, hidden_action_fc)
        self.mu_output = torch.nn.Linear(hidden_layers_size[-1], self.action_dim)
        self.log_std_output = torch.nn.Linear(hidden_layers_size[-1], self.action_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.linear_mlp_stack(x)
        mu = self.mu_output(x)
        log_std = self.log_std_output(x)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        return mu, std

    def act(self, x: torch.Tensor, stochastically=True, with_log_pro=False):
        device = x.device
        self.action_mean = self.action_mean.to(device)
        self.action_radius = self.action_radius.to(device)
        self.action_low = self.action_low.to(device)
        self.action_high = self.action_high.to(device)
        mu, std = self.forward(x)
        mu = mu + self.action_mean
        log_pro = torch.zeros(mu[0]).resize([-1, 1])
        if stochastically:
            action = torch.randn_like(mu) * std + mu
            action = torch.clamp(action, min=self.action_low, max=self.action_high)
            if with_log_pro:
                log_pro = LOG_GAUSSIAN_NORMAL_CONSTANT - torch.log(std) - \
                          0.5 * (action-mu)*(action-mu).sum(axis=-1, keepdims=True)
            return action, log_pro
        else:
            return mu, log_pro


class MLPGaussianActorSquashing(MLPGaussianActor):
    def __init__(self, observation_space:gym.Space,
                 action_space: gym.Space,
                 hidden_layers_size: list,
                 hidden_action_fc):
        super(MLPGaussianActorSquashing,
              self).__init__(observation_space, action_space,
                             hidden_layers_size, hidden_action_fc)

    def act(self, x: torch.Tensor, stochastically=True, with_log_pro=False):
        mu, std = self.forward(x)
        log_pro = None
        device = x.device
        self.action_mean = self.action_mean.to(device)
        if stochastically:
            self.action_radius = self.action_radius.to(device)
            action_noise = torch.randn_like(mu)
            actions = action_noise * std + mu
            actions_squashing = torch.tanh(actions) * self.action_radius + self.action_mean
            if with_log_pro:
                # da_du = torch.sum(torch.log(1. - torch.tanh(actions) ** 2), dim=1, keepdim=True)
                da_du = torch.sum(2 * np.log(2) - 2 * torch.log(torch.exp(2 * actions + 1) - actions),
                                  dim=1, keepdim=True)
                gaussian_normalize_0 = len(mu[-1]) * LOG_GAUSSIAN_NORMAL_CONSTANT
                gaussian_normalize_1 = 0.5 * torch.log((std * std).sum(dim=1, keepdim=True))
                gaussian_exp = 0.5 * (action_noise * action_noise).sum(dim=1, keepdim=True)
                log_pro = gaussian_normalize_0 - gaussian_normalize_1 - gaussian_exp - da_du
                # log_pro = LOG_GAUSSIAN_NORMAL_CONSTANT - torch.log(std) - 0.5 * action_noise * action_noise
                # log_pro -= torch.log((1 - actions_squashing * actions_squashing) + 1e-6)
                # log_pro = log_pro.sum(1, keepdim=True)
            return actions_squashing, log_pro
        else:
            return mu + self.action_mean, log_pro


class MLPGaussianActorManuSTD(Actor):
    def __init__(self, observation_space: gym.Space,
                 action_space: gym.Space,
                 hidden_layers_size: list,
                 hidden_action, output_action,
                 std_init=0.1,
                 std_decay=1.0):
        """

        :param obs_dim:
        :param action_dim:
        :param hidden_layers_size:
        :param hidden_action:
        :param output_action:
        :param std_init:
        :param std_decay:
        """
        super(MLPGaussianActorManuSTD, self).__init__(observation_space, action_space)
        layers = [self.obs_dim, *hidden_layers_size, self.action_dim]
        self.linear_mlp_stack = MLP(layers,  hidden_action, output_action)
        self.std = torch.tensor(std_init, dtype=torch.float32)
        self.std_decay = std_decay

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu = self.linear_mlp_stack(x)
        mu = mu
        return mu,  self.std

    def update_std(self):
        self.std = self.std * self.std_decay

    def act(self, x: torch.Tensor, stochastically=True, with_log_pro=False):
        mu, std = self.forward(x)
        device = x.device
        mu = mu + self.action_mean.to(device)
        log_pro = None
        if stochastically:
            action_noise = torch.randn_like(mu)
            actions = action_noise * std + mu
            actions = torch.clamp(actions, min=self.action_low.to(device), max=self.action_high.to(device))
            if with_log_pro:
                log_pro = LOG_GAUSSIAN_NORMAL_CONSTANT - np.log(std) - \
                          0.5 * (actions - mu) * (actions - mu).sum(axis=-1, keepdims=True)
            return actions, log_pro

        else:
            return mu, log_pro


class MLPCritic(Critic):
    def __init__(self, observation_space: gym.Space,
                 action_space: gym.Space,
                 hidden_layers_size: list, hidden_action):
        super(MLPCritic, self).__init__(observation_space, action_space)
        layers = [self.obs_dim + self.action_dim, *hidden_layers_size, 1]
        self.linear_stack = MLP(layers, hidden_action)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), 1)
        output = self.linear_stack(x)
        return output
