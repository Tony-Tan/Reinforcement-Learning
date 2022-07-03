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
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplement

    def distribution(self, x: torch.Tensor):
        raise NotImplement

    def act(self, x: torch.Tensor, stochastically=True):
        raise NotImplement

    def act_pro(self, x: torch.Tensor) -> tuple:
        raise NotImplement


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        pass

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
        self.start_epoch = 1
        self.path = path
        self.model_name_ = None
        self.hyperparameter = Hyperparameter(path)

    def reaction(self, state: np.ndarray):
        raise NotImplement

    def simulation(self):
        raise NotImplement

    def save(self, epoch_num):
        self.model_name_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        policy_module_path = os.path.join(self.path, self.model_name_) + '_actor.pt'
        torch.save(self.actor, policy_module_path)
        value_module_path = os.path.join(self.path, self.model_name_) + '_critic.pt'
        torch.save(self.critic, value_module_path)
        model_rec_file = open(os.path.join(self.path, 'last_models.txt'), 'w+')
        model_rec_file.write(self.model_name_ + '\n')
        model_rec_file.write(str(epoch_num) + '\n')
        model_rec_file.close()
        print('model saved! ')

    def load(self, model_name=None):
        self.model_name_ = model_name
        if self.model_name_ is None:
            model_name_file_path = os.path.join(self.path, 'last_models.txt')
            if os.path.exists(model_name_file_path):
                model_rec_file = open(model_name_file_path, 'r')
                self.model_name_ = model_rec_file.readline().strip('\n')
                self.start_epoch = int(model_rec_file.readline().strip('\n')) + 1
                model_rec_file.close()
            else:
                return
        actor_module_path = os.path.join(self.path, self.model_name_) + '_actor.pt'
        self.actor = torch.load(actor_module_path, map_location=torch.device('cpu'))
        critic_module_path = os.path.join(self.path, self.model_name_) + '_critic.pt'
        self.critic = torch.load(critic_module_path, map_location=torch.device('cpu'))
        print('================================================================')
        print('model loaded: ' + self.model_name_)
        print('================================================================')

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
        current_state = self.env.reset()
        while step_num < total_step_num:
            action = self.agent.actor.act(current_state)
            new_state, reward, is_done, _ = self.env.step(action)
            self.buffer.push([current_state, action, reward, is_done])
            step_num += 1
            if is_done:
                current_state = self.env.reset()
            else:
                current_state = new_state

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
    def __init__(self, state_dim, action_dim,
                 hidden_layers_size: list,
                 hidden_action_fc,
                 output_action_fc: dict):
        """
        :param state_dim:
        :param action_dim:
        :param hidden_layers_size:
        :param hidden_action_fc:
        :param output_action_fc: dict, {'mu':action, 'std':action}
        """
        super(MLPGaussianActor, self).__init__()
        layers = [state_dim, *hidden_layers_size]
        self.linear_mlp_stack = MLP(layers, hidden_action_fc, hidden_action_fc)
        self.mu_output = torch.nn.Sequential(torch.nn.Linear(hidden_layers_size[-1], action_dim))
        self.std_output = torch.nn.Sequential(torch.nn.Linear(hidden_layers_size[-1], action_dim),
                                              torch.nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.linear_mlp_stack(x)
        mu = self.mu_output(x)
        std = self.std_output(x)
        # std = torch.clip(std, min=-20, max=-2.5)
        # std = torch.exp(std)
        std = std * 0.2
        return mu, std

    def distribution(self, x: torch.Tensor):
        return self.forward(x)

    def act(self, x: torch.Tensor, stochastically=True):
        mu, std = self.forward(x)
        if stochastically:
            return torch.randn_like(mu)*std+mu
        else:
            return mu

    def act_pro(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu, std = self.forward(x)
        action_noise = torch.randn_like(mu)
        actions = action_noise*std+mu
        action_pro = GAUSSIAN_NORMAL_CONSTANT/std * torch.exp(-0.5 * action_noise*action_noise)
        return actions, action_pro


class MLPGaussianActorSquashing(MLPGaussianActor):
    def __init__(self, state_dim, action_dim,
                 hidden_layers_size: list,
                 hidden_action_fc,
                 output_action_fc: dict
                 ):
        super(MLPGaussianActorSquashing,
              self).__init__(state_dim, action_dim,
                             hidden_layers_size, hidden_action_fc, output_action_fc)

    def distribution(self, x: torch.Tensor):
        return self.forward(x)

    def act(self, x: torch.Tensor, stochastically=True):
        action_raw = super(MLPGaussianActorSquashing, self).act(x, stochastically)
        return torch.tanh(action_raw)

    def act_logpro(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu, std = self.forward(x)
        action_noise = torch.randn_like(mu)
        actions = action_noise*std+mu
        actions_squashing = torch.tanh(actions)
        da_du = torch.sum(torch.log(1 - torch.tanh(actions) ** 2), dim=1, keepdim=True)
        log_pro = LOG_GAUSSIAN_NORMAL_CONSTANT - torch.log(std) - 0.5 * action_noise * action_noise - da_du
        return actions_squashing, log_pro

    def act_pro(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        actions, log_pro = self.act_logpro(x)
        return actions, torch.exp(log_pro)


class MLPGaussianActorManuSTD(Actor):
    def __init__(self, state_dim, action_dim,
                 hidden_layers_size: list,
                 hidden_action, output_action,
                 mu_output_shrink,
                 std_init: float,
                 std_decay: float):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers_size:
        :param hidden_action:
        :param output_action:
        :param std_init:
        :param std_decay:
        """
        super(MLPGaussianActorManuSTD, self).__init__()
        self.mu_output_shrink = mu_output_shrink
        layers = [state_dim, *hidden_layers_size, action_dim]
        self.linear_mlp_stack = MLP(layers,  hidden_action, output_action)
        self.std = std_init
        self.std_decay = std_decay

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu = self.linear_mlp_stack(x)
        mu = mu * self.mu_output_shrink
        return mu,  self.std

    def update_std(self):
        self.std = self.std * self.std_decay

    def distribution(self, x: torch.Tensor):
        return self.forward(x)

    def act(self, x: torch.Tensor, stochastically=True):
        mu, std = self.forward(x)
        if stochastically:
            return torch.randn_like(mu) * std + mu
        else:
            return mu

    def act_pro(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu, std = self.forward(x)
        actions = torch.randn_like(mu)*std+mu
        action_pro = GAUSSIAN_NORMAL_CONSTANT/std * torch.exp(-0.5 * torch.pow((actions-mu)/std, 2))
        return actions, action_pro


class MLPCritic(Critic):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers_size: list, hidden_action):
        super(MLPCritic, self).__init__()
        layers = [state_dim+action_dim, *hidden_layers_size, 1]
        self.linear_stack = MLP(layers, hidden_action)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), 1)
        output = self.linear_stack(x)
        return output
