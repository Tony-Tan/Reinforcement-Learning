import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import environments.tetris as tetris
from torch.utils.tensorboard import SummaryWriter
import environments.generate_trajectories_set as gts
import environments.print_time as pt
import os
import time
from multiprocessing import Pool

DEBUG_FLAG = False
TETRIS_WIDTH = 5
TETRIS_HEIGHT = 8
writer = SummaryWriter('./data/log/')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


class policyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(policyNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        p = self.linear_relu_stack(x)
        return p


def conjugate_gradient(A, b, max_step_num):
    with torch.no_grad():
        x_k = torch.zeros(b.shape).to(device)
        # r_k = torch.matmul(A, x_k) - b
        r_k = - b
        p_k = b
        for k_i in range(max_step_num):
            if r_k.all() == 0:
                break
            alpha_k = torch.matmul(r_k.t(), r_k) / torch.matmul(torch.matmul(p_k.t(), A), p_k)
            x_k = x_k + alpha_k * p_k
            r_k_1 = r_k + alpha_k * torch.matmul(A, p_k)
            beta_k_1 = torch.matmul(r_k_1.t(), r_k_1) / torch.matmul(r_k.t(), r_k)
            p_k = - r_k_1 + beta_k_1 * p_k
            r_k = r_k_1
        return x_k


class TRPO_Agent:
    def __init__(self, env_, delta_limit_, module_path=None, method_='TRPO'):
        self.method = method_
        self.env = env_
        self.input_size = len(self.env.reset())
        self.action_space_size = len(self.env.action_space)
        self.delta_limit = torch.tensor(delta_limit_).to(device)
        self.trajectory_num_per_update = 256

        self.weights_state_value = None
        # self.weights_policy = torch.zeros([self.action_space_size, input_size_], requires_grad=True,
        #                                   dtype=torch.float32)
        self.policy_module = policyNN(self.input_size, self.action_space_size).to(device)
        self.policy_parameter_size = 0
        for p in self.policy_module.parameters():
            self.policy_parameter_size += p.data.numel()
            # torch.zero_(p.data)
        if module_path is not None:
            self.policy_module = torch.load(module_path)
            self.policy_module.eval()
        pass

    def save_model(self, path='./data/models/'):
        model_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        model_path = os.path.join(path, model_name) + '.pt'
        torch.save(self.policy_module, model_path)
        print('model saved: ' + model_path)

    # def policy(self, x_tensor):
    #     prob = F.softmax(torch.sum(torch.mul(x_tensor, self.weights_policy), dim=1), dim=0).requires_grad_()
    #     prob = torch.add(torch.mul(prob, 0.9999), 0.0001 / self.action_space_size)
    #     return prob

    def policy(self, x_tensor):
        prob = self.policy_module(x_tensor)
        return prob

    def policy_numpy(self, x_tensor):
        with torch.no_grad():
            prob_np = self.policy(x_tensor)
        return prob_np.cpu().detach().numpy().flatten()

    def state_value(self, x_tensor):
        x_tensor = torch.cat((x_tensor, torch.ones([1, 1], dtype=torch.float32)), dim=1)
        state_value = torch.matmul(x_tensor, self.weights_state_value)
        return state_value

    def select_action(self, x_tensor):
        policy = self.policy(x_tensor).clone().detach().numpy()
        action = np.random.choice(range(self.action_space_size), 1, p=policy)[0]
        return action

    def shrink_update(self, fim, x_k):
        delta_theta_k = torch.sqrt(2 * self.delta_limit / (torch.matmul(torch.matmul(x_k.t(), fim), x_k))) * x_k

        divergence = torch.matmul(torch.matmul(delta_theta_k.t(), fim), delta_theta_k)
        while divergence >= self.delta_limit:
            delta_theta_k *= 0.5
            divergence = torch.matmul(torch.matmul(delta_theta_k.t(), fim), delta_theta_k)
        return delta_theta_k

    def info_print(self, epoch_i, log_average_step_num, log_reward, log_inter=100.):
        self.save_model()
        pt.print_time()
        writer.add_scalar('total step', log_average_step_num / (self.trajectory_num_per_update * log_inter),
                          self.trajectory_num_per_update * epoch_i)
        writer.add_scalar('reward', log_reward / (self.trajectory_num_per_update * 100.),
                          self.trajectory_num_per_update * epoch_i)
        print('total step: ' + str(log_average_step_num / (self.trajectory_num_per_update * log_inter)))
        print('reward: ' + str(log_reward / (self.trajectory_num_per_update * log_inter)))
        print('policy weights:')
        for p_i in self.policy_module.parameters():
            print(p_i.data)
        print('------------------------------------------------------------')

    def generate_trajectory_set(self, set_size):
        trajectory_and_reward_array = []
        for i in range(set_size):
            total_reward = 0
            trajectory_i = []
            current_state = self.env.reset().reshape([1, -1])
            is_done = False
            while not is_done:
                current_state_ts = torch.tensor(current_state, dtype=torch.float32,
                                                requires_grad=False).to(device)
                policy_pdf_tensor = self.policy(current_state_ts)
                policy_pdf = policy_pdf_tensor.cpu().detach().numpy().flatten()
                action = np.random.choice(range(self.action_space_size), 1, p=policy_pdf)[0]
                action_lh = policy_pdf[action]
                next_state, reward, is_done, _ = self.env.step(action)
                next_state = next_state.reshape([1, -1])
                # next_state_feature = torch.tensor(feature_fn(next_state), dtype=torch.float32,
                #                                   requires_grad=False).to(device)
                trajectory_i.append([current_state, action, reward, action_lh])
                current_state = next_state
                total_reward += reward
            if total_reward == 0:
                i -= 1
                continue
            trajectory_and_reward_array.append([trajectory_i, total_reward])
        return trajectory_and_reward_array

    def optimization(self, epoch_size, value_gamma=0.9, value_step_size=.1, conjugate_step_k=10, thread_num=8):
        log_reward = 0
        log_average_step_num = 0
        baseline = 0
        log_inter = 10
        for epoch_i in range(1, epoch_size):
            # print log
            if epoch_i % log_inter == 0:
                self.info_print(epoch_i, log_average_step_num, log_reward, log_inter)
                baseline = log_reward / (self.trajectory_num_per_update * log_inter)
                log_reward = 0
                log_average_step_num = 0
            # collect set of N trajectories
            trajectory_and_reward_collection = self.generate_trajectory_set(self.trajectory_num_per_update)
            total_trajectory_step_num = 0
            for t_r_i in trajectory_and_reward_collection:
                trajectory_i = t_r_i[0]
                return_value = t_r_i[1]
                trajectory_step_num = len(trajectory_i)
                total_trajectory_step_num += trajectory_step_num
                log_average_step_num += trajectory_step_num
                log_reward += return_value
                n = len(trajectory_i)
                trajectory_i[n - 1].append(np.float32(trajectory_i[n - 1][2]))
                for step_index in reversed(range(n)):
                    if step_index + 1 < n:
                        trajectory_i[step_index].append(np.float32(trajectory_i[step_index][2]) +
                                                        value_gamma * np.float32(trajectory_i[step_index + 1][-1]))

            # trpo
            fisher_matrix = torch.zeros([self.policy_parameter_size, self.policy_parameter_size],
                                        dtype=torch.float32).to(device)
            eta_summation = torch.zeros([self.policy_parameter_size, 1],
                                        dtype=torch.float32).to(device)
            for t_r_i in trajectory_and_reward_collection:
                trajectory_i = t_r_i[0]
                trajectory_step_num = len(trajectory_i)
                # -----------------------------------
                # for step_index in range(len(trajectory_i) - 1):
                for step_index in range(trajectory_step_num):
                    # build up the input tensor
                    current_x_tensor = torch.tensor(trajectory_i[step_index][0], dtype=torch.float32,).to(device)
                    q_value = trajectory_i[step_index][4] - baseline
                    prob = self.policy(current_x_tensor)
                    prob_num = prob.clone().detach()
                    action = trajectory_i[step_index][1]
                    for action_i in range(self.action_space_size):
                        output = torch.zeros_like(prob).to(device)
                        # output[0][action_i] = 1
                        self.policy_module.zero_grad()
                        prob.backward(output, retain_graph=True)
                        gradient = []
                        for p_i in self.policy_module.parameters():
                            if not p_i.requires_grad:
                                continue
                            gradient.append(p_i.grad.reshape([-1, 1]))

                        gradient = torch.cat(gradient, dim=0).to(device)
                        log_gradient = gradient / prob_num[0][action_i]
                        if action == action_i:
                            eta_summation += (log_gradient * q_value)
                        fisher_matrix += torch.matmul(log_gradient, log_gradient.t()) * prob_num[0][action_i]

            with torch.no_grad():
                if torch.count_nonzero(eta_summation) == 0:
                    continue
                # total_step_num = torch.tensor([total_step_num], dtype=torch.float32).to(device)
                gradient_estimate = eta_summation / total_trajectory_step_num
                fim = fisher_matrix / total_trajectory_step_num + torch.eye(gradient_estimate.size()[0]).to(
                    device) * 1e-3
                if self.method == 'TRPO':
                    # conjugate algorithm first K step
                    # Hx = g
                    # x = H^{-1}g
                    x_k = conjugate_gradient(fim, gradient_estimate, conjugate_step_k)
                    if torch.any(torch.isnan(x_k)):
                        print('x_k contains nan')
                        continue
                    delta_weight = self.shrink_update(fim, x_k)
                    if torch.any(torch.isnan(delta_weight)):
                        print('delta weight contains nan')
                        continue
                    array_update_position = 0
                    for p_i in self.policy_module.parameters():
                        if not p_i.requires_grad:
                            continue
                        delta_p_i = delta_weight[array_update_position: array_update_position + p_i.data.numel()]
                        array_update_position += p_i.data.numel()
                        delta_p_i = torch.tensor(delta_p_i, dtype=torch.float32).resize_as(p_i.data)
                        p_i.data += delta_p_i

                # elif self.method == 'NPG':
                #     delta_weight = np.linalg.inv(fim).dot(eta_summation)
                #     delta_weight /= (10 * np.max(delta_weight))
                #     new_weight = self.weights_policy.clone().detach().numpy() + \
                #                  delta_weight.reshape(self.action_space_size, -1)
                #     self.weights_policy = torch.tensor(new_weight, requires_grad=True, dtype=torch.float32)
            # if DEBUG_FLAG:
            #     print('==============================================================================================')
            # -----------------------------------


if __name__ == '__main__':
    env = tetris.Tetris(TETRIS_WIDTH, TETRIS_HEIGHT)
    trpo_agent = TRPO_Agent(env, delta_limit_=0.01, method_='TRPO')
    trpo_agent.optimization(100000)
