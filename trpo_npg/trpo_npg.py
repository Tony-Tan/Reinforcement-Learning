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

DEBUG_FLAG = False
TETRIS_WIDTH = 5
TETRIS_HEIGHT = 6
writer = SummaryWriter('./data/log/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def features(state):
    state_bg = state[0].reshape(1, -1).astype(np.float32)
    # heights_arr = (state[0] != 0).argmax(axis=0) / tetris_height
    # diff_height = np.array(
    #     [heights_arr[i] - heights_arr[i + 1] for i in range(len(heights_arr) - 1)]) / tetris_height
    # the last number is for bias
    state_t = np.array([state[1], state[2] / 360., state[3][0] / TETRIS_WIDTH,
                        state[3][1] / TETRIS_HEIGHT]).astype(np.float32)
    # state_ = np.append(heights_arr, diff_height)
    # state_ = np.append(state_, state_t)
    state_ = np.append(state_bg, state_t).reshape([1, -1])
    x_tensor_current = state_
    return x_tensor_current


class policyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(policyNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 5, bias=False),
            nn.Tanh(),
            nn.Linear(5, output_size, bias=False),
            nn.Tanh(),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.flatten(x)
        p = self.linear_relu_stack(x)
        # p = p * 0.999 + 0.001 / torch.numel(p)
        return p


def conjugate_gradient(A, b, max_step_num):
    with torch.no_grad():
        x_k = torch.randn(b.shape).to(device)/2.
        r_k = torch.matmul(A, x_k) - b
        p_k = - r_k
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
    def __init__(self, env_, delta_limit_, input_size_, module_path=None, method_='TRPO'):
        self.method = method_
        self.env = env_
        # self.input_size = input_size_
        self.action_space_size = len(self.env.action_space)
        self.delta_limit = torch.tensor(delta_limit_).to(device)
        self.trajectory_num_per_update = 400

        self.weights_state_value = None
        # self.weights_policy = torch.zeros([self.action_space_size, input_size_], requires_grad=True,
        #                                   dtype=torch.float32)
        self.policy_module = policyNN(input_size_, self.action_space_size).to(device)
        self.policy_parameter_size = 0
        for p in self.policy_module.parameters():
            self.policy_parameter_size += p.data.numel()
            # torch.zero_(p.data)
        if module_path is not None:
            self.policy_module = torch.load(module_path)
            self.policy_module.eval()
        pass

    def save_model(self, path='./data/models/'):
        model_name = time.strftime("%H-%M-%S", time.localtime())
        model_path = os.path.join(path, model_name)+'.pt'
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
        x_tensor = torch.cat((x_tensor, torch.ones([1,1], dtype=torch.float32)),dim=1)
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

    def optimization(self, epoch, value_gamma=0.9, value_step_size=.1, conjugate_step_k=10):
        log_reward = 0
        total_step_num = 0
        for epoch_i in range(1, epoch):
            # print log
            if epoch_i % 100 == 0:
                self.save_model()
                pt.print_time()
                writer.add_scalar('total step', total_step_num, self.trajectory_num_per_update * epoch_i)
                writer.add_scalar('reward', log_reward / (self.trajectory_num_per_update * 100.),
                                  self.trajectory_num_per_update * epoch_i)
                print('total step: ' + str(total_step_num))
                print('reward: ' + str(log_reward / (self.trajectory_num_per_update * 100.)))
                print('policy weights:')
                for p_i in self.policy_module.parameters():
                    print(p_i.data)
                print('------------------------------------------------------------')
                log_reward = 0

            fisher_matrix = torch.zeros([self.policy_parameter_size, self.policy_parameter_size],
                                        dtype=torch.float32).to(device)
            eta_summation = torch.zeros([self.policy_parameter_size, 1],
                                        dtype=torch.float32).to(device)
            total_step_num = 0
            delta_weight = None
            # collect set of N trajectories
            trajectory_collection, reward_sum = gts.generate_trajectory_set(self.env, self.trajectory_num_per_update,
                                                                            features, max_trajectory_length=1000,
                                                                            device=device, policy_fn=self.policy_numpy)
            if trajectory_collection is None:
                print('Nan in policy and its weights')
                for p_i in self.policy_module.parameters():
                    print(p_i.data)
            log_reward += reward_sum
            # regress the state-value linear estimate
            # data_collection = []  # least square methods for regressing state value
            # labels_collection = []  # least square methods for regressing state value
            for trajectory_i in trajectory_collection:
                n = len(trajectory_i)
                for step_index in reversed(range(n)):
                    if step_index + 1 < n:
                        trajectory_i[step_index][2] = (trajectory_i[step_index][2] +
                                                       value_gamma * trajectory_i[step_index + 1][2])
                        # labels_collection.append(labels_collection[-1]+trajectory_i[step_index][2])
                    # else:
                    #     labels_collection.append(trajectory_i[step_index][2])
                    # data_collection.append(np.array(trajectory_i[step_index][0][0]))
            #
            # data_collection = np.array(data_collection)
            # data_collection = np.hstack([data_collection, np.ones([len(labels_collection), 1])])
            # labels_collection = np.array(labels_collection).reshape(-1, 1)
            # weights = np.linalg.lstsq(data_collection, labels_collection, rcond=None)[0]
            # self.weights_state_value = torch.tensor(weights, requires_grad=False, dtype=torch.float32)
            # trpo
            for trajectory_i in trajectory_collection:
                for step_index in range(len(trajectory_i) - 1):
                    # build up the input tensor
                    current_x_tensor = trajectory_i[step_index][0]
                    next_x_tensor = trajectory_i[step_index + 1][0]
                    # q_value = trajectory_i[step_index][2] + self.state_value(
                    #     next_x_tensor) - self.state_value(current_x_tensor)
                    q_value = trajectory_i[step_index][2]
                    prob = self.policy(current_x_tensor)
                    prob_num = prob.clone().detach()
                    action = trajectory_i[step_index][1]
                    for action_i in range(self.action_space_size):
                        output = torch.zeros_like(prob).to(device)
                        output[0][action_i] = 1
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
                            eta_summation += (log_gradient * q_value )
                        fisher_matrix += torch.matmul(log_gradient, log_gradient.t()) * prob_num[0][action_i]
                        # if torch.any(torch.isnan(fisher_matrix)):
                        #     print('gradient:')
                        #     print(gradient.t())
                        #     print('probability:')
                        #     print(prob_num)
                    total_step_num += 1
            with torch.no_grad():
                if torch.count_nonzero(eta_summation) == 0:
                    continue
                # total_step_num = torch.tensor([total_step_num], dtype=torch.float32).to(device)
                gradient_estimate = eta_summation / total_step_num
                if DEBUG_FLAG:
                    print('gradient estimate:')
                    print(gradient_estimate.t())
                fim = fisher_matrix / total_step_num + torch.eye(gradient_estimate.size()[0]).to(device) * 1e-3
                if DEBUG_FLAG:
                    print('fisher information matrix:')
                    print(fim)
                    print('total step numbers:')
                    print(total_step_num)
                if self.method == 'TRPO':
                    # conjugate algorithm first K step
                    # Hx = g
                    # x = H^{-1}g
                    x_k = conjugate_gradient(fim, gradient_estimate, conjugate_step_k)
                    if torch.any(torch.isnan(x_k)):
                        print('x_k contains nan')
                        continue
                    if DEBUG_FLAG:
                        print('delta weights before shrink')
                        print(x_k.t())
                    # backtracking line search
                    # theta_k = self.weights_policy.clone().detach().numpy().reshape((-1, 1))
                    delta_weight = self.shrink_update(fim, x_k)
                    if torch.any(torch.isnan(delta_weight)):
                        print('delta weight contains nan')
                        continue
                    if DEBUG_FLAG:
                        print('delta weights after shrink:')
                        print(delta_weight.t())
                    # update
                    if DEBUG_FLAG:
                        print('policy weights before update')
                        for p_i in self.policy_module.parameters():
                            print(p_i.data)
                    # -----------------------------------------------------------
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


            if DEBUG_FLAG:
                print('==============================================================================================')
            # -----------------------------------


if __name__ == '__main__':
    env = tetris.Tetris(TETRIS_WIDTH, TETRIS_HEIGHT)
    trpo_agent = TRPO_Agent(env, delta_limit_=0.001, input_size_=TETRIS_WIDTH * TETRIS_HEIGHT + 4, method_='TRPO')
    trpo_agent.optimization(100000)
