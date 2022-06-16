import numpy as np
import torch
import torch.nn as nn
import time
import os
from core.exceptions import *
from core.basic import *
import json
from torch.utils.tensorboard import SummaryWriter


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self, x: torch.Tensor):
        raise NotImplement

    def _distribution(self, state: np.ndarray):
        raise NotImplement


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        pass

    def forward(self, x: torch.Tensor):
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
        self.hyperparameter = {}
        self.path = path
        self.load()

    def reaction(self, state: np.ndarray):
        raise NotImplement

    def simulation(self):
        raise NotImplement

    def save_hyperparameter(self):
        hyperparameter_json = json.dumps(self.hyperparameter)
        with open(os.path.join(self.path, 'hyperparameter.json'), 'w') as j_file:
            j_file.write(hyperparameter_json)
            j_file.close()
            print('hyperparameter saved')

    def load_hyperparameter(self):
        with open(os.path.join(self.path, 'hyperparameter.json'), 'r') as j_file:
            content = j_file.read()
            self.hyperparameter = json.loads(content)
            j_file.close()
            print('hyperparameter loaded')

    def save(self, epoch_num=0):
        self.save_hyperparameter()
        model_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        policy_module_path = os.path.join(self.path, model_name) + '_actor.pt'
        torch.save(self.actor, policy_module_path)
        value_module_path = os.path.join(self.path, model_name) + '_critic.pt'
        torch.save(self.critic, value_module_path)
        model_rec_file = open(os.path.join(self.path, 'last_models.txt'), 'w+')
        model_rec_file.write(model_name + '\n')
        model_rec_file.write(str(epoch_num) + '\n')
        model_rec_file.close()
        print('model saved! ')

    def load(self):

        model_name_file_path = os.path.join(self.path, 'last_models.txt')
        if os.path.exists(model_name_file_path):
            self.load_hyperparameter()
            model_rec_file = open(model_name_file_path, 'r')
            model_name = model_rec_file.readline().strip('\n')
            self.start_epoch = int(model_rec_file.readline().strip('\n')) + 1
            model_rec_file.close()
            actor_module_path = os.path.join(self.path, model_name) + '_actor.pt'
            self.actor = torch.load(actor_module_path)
            critic_module_path = os.path.join(self.path, model_name) + '_critic.pt'
            self.critic = torch.load(critic_module_path)
            print('================================================================')
            print('model loaded: ' + model_name)
            print('================================================================')

    def update_actor(self, epoch_num: int, data: DataBuffer, device: str, log_writer: SummaryWriter):
        raise NotImplement

    def update_critic(self, epoch_num: int, data: DataBuffer, device: str, log_writer: SummaryWriter):
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
            action = self.agent.reaction(current_state)
            new_state, reward, is_done, _ = self.env.step(action)
            self.buffer.push([current_state, action, reward, is_done])
            step_num += 1
            if is_done:
                current_state = self.env.reset()
            else:
                current_state = new_state

    def play(self, round_num: int):
        raise NotImplement

    def test(self, round_num: int, test_round_num: int, device='cpu'):
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

    def save(self):
        self.agent.save()
