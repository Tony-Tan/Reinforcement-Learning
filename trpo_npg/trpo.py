import numpy as np
import torch
import torch.nn.functional as F
import environments.simplified_tetris as tetris
from torch.utils.tensorboard import SummaryWriter

TETRIS_WIDTH = 6
TETRIS_HEIGHT = 5
writer = SummaryWriter('./data/log/')


def features(state):
    state_bg = (state[0].reshape(1, -1).astype(np.float32) - 0.5) * 2
    # heights_arr = (state[0] != 0).argmax(axis=0) / tetris_height
    # diff_height = np.array(
    #     [heights_arr[i] - heights_arr[i + 1] for i in range(len(heights_arr) - 1)]) / tetris_height
    # the last number is for bias
    state_t = np.array([state[1], state[2] / 360., state[3][0] / TETRIS_WIDTH,
                        state[3][1] / TETRIS_HEIGHT, 1.0]).astype(np.float32)
    # state_ = np.append(heights_arr, diff_height)
    # state_ = np.append(state_, state_t)
    state_ = np.append(state_bg, state_t)
    x_tensor_current = torch.from_numpy(state_)
    return x_tensor_current


class TRPO_Agent:
    def __init__(self, env_, delta_limit_, input_size_,method_='TRPO'):
        self.method = method_
        self.env = env_
        # self.input_size = input_size_
        self.action_space_size = len(self.env.action_space)
        self.delta_limit = delta_limit_
        self.trajectory_num_per_update = 100

        self.weights_state_value = None
        self.weights_policy = torch.zeros([self.action_space_size, input_size_], requires_grad=True,dtype=torch.float32)
        pass

    def policy(self, x_tensor):
        prob = F.softmax(torch.sum(torch.mul(x_tensor, self.weights_policy), dim=1), dim=0).requires_grad_()
        prob = torch.add(torch.mul(prob, 0.9999), 0.0001 / self.action_space_size)
        return prob

    def state_value(self, x_tensor):
        state_value = torch.sum(torch.mul(x_tensor, self.weights_state_value), dim=0)
        return state_value

    def select_action(self, x_tensor):
        policy = self.policy(x_tensor).clone().detach().numpy()
        action = np.random.choice(range(self.action_space_size), 1, p=policy)[0]
        return action

    def optimization(self, epoch, value_gamma=0.9, value_step_size=.1, conjugate_step_k=10):
        log_reward = 0
        for epoch_i in range(1, epoch):

            trajectory_collection = []
            fisher_matrix = 0
            eta_summation = torch.zeros_like(self.weights_policy,dtype=torch.float32)
            total_step_num = 0
            delta_weight = None
            # collect set of N trajectories
            for n_i in range(self.trajectory_num_per_update):
                trajectory_i = []
                current_state, reward, is_done, _ = self.env.reset()
                current_state_feature = features(current_state)
                while not is_done:
                    # generate experience

                    action = self.select_action(current_state_feature)
                    next_state, reward, is_done, _ = self.env.step_autofill(action)
                    next_state_feature = features(next_state)
                    trajectory_i.append([current_state_feature, action, reward])
                    current_state_feature = next_state_feature
                    # log
                    log_reward += reward
                trajectory_collection.append(trajectory_i)

            # regress the state-value linear estimate
            data_collection = []  # least square methods for regressing state value
            labels_collection = []  # least square methods for regressing state value
            for trajectory_i in trajectory_collection:
                n = len(trajectory_i)
                for step_index in reversed(range(n)):
                    if step_index+1 < n:
                        trajectory_i[step_index][2] = (trajectory_i[step_index][2]+trajectory_i[step_index+1][2])
                    # else:
                    #     labels_collection.append(trajectory_i[step_index][2])
                    # data_collection.append(trajectory_i[step_index][0].clone().detach().numpy())

            # data_collection = np.array(data_collection)
            # data_collection = np.hstack([data_collection, np.ones([len(labels_collection), 1])])
            # labels_collection = np.array(labels_collection).reshape(-1, 1)
            # weights = np.linalg.lstsq(data_collection, labels_collection, rcond=None)[0]
            # self.weights_state_value = torch.tensor(weights, requires_grad=False,dtype=torch.float32)


            # trpo
            for trajectory_i in trajectory_collection:
                for step_index in range(len(trajectory_i) - 1):
                    # build up the input tensor
                    current_x_tensor = trajectory_i[step_index][0]
                    next_x_tensor = trajectory_i[step_index + 1][0]
                    # q_value = trajectory_i[step_index][2] + value_gamma * self.state_value(
                    #     next_x_tensor)  # - self.state_value(current_x_tensor)
                    q_value = trajectory_i[step_index][2]
                    prob = self.policy(current_x_tensor)
                    prob_num = prob.clone().detach().numpy()
                    action = trajectory_i[step_index][1]
                    for action_i in range(self.action_space_size):
                        gradient = \
                        torch.autograd.grad(outputs=prob[action_i], inputs=self.weights_policy, retain_graph=True)[0]
                        log_gradient = gradient / prob[action_i]
                        if action == action_i:
                            eta_summation = eta_summation + log_gradient * q_value
                        # Fisher Information Estimate
                        log_gradient = log_gradient.clone().detach().numpy().reshape((1, -1))
                        fisher_matrix += (prob_num[action_i] * log_gradient.transpose().dot(log_gradient))
                    total_step_num += 1

            eta_summation = eta_summation.clone().detach().numpy().reshape((-1, 1))
            if np.all(eta_summation == 0):
                continue
            gradient_estimate = eta_summation / total_step_num

            fim = fisher_matrix / total_step_num + np.eye(np.shape(fisher_matrix)[0]) * 0.001
            if self.method == 'TRPO':
                # conjugate algorithm first K step
                # Hx = g
                # x = H^{-1}g

                x_k = np.random.random(np.shape(gradient_estimate)) - 0.5
                r_k = np.dot(fim, x_k) - gradient_estimate
                p_k = - r_k
                for k_i in range(conjugate_step_k):
                    if r_k.all() == 0:
                        break
                    alpha_k = np.transpose(r_k).dot(r_k) / (np.transpose(p_k).dot(fim).dot(p_k))
                    x_k = x_k + alpha_k * p_k
                    r_k_1 = r_k + alpha_k * np.dot(fim, p_k)
                    beta_k_1 = r_k_1.transpose().dot(r_k_1) / r_k.transpose().dot(r_k)
                    p_k = - r_k_1 + beta_k_1 * p_k
                    r_k = r_k_1

                # backtracking line search
                # theta_k = self.weights_policy.clone().detach().numpy().reshape((-1, 1))
                delta_theta_k = np.sqrt(2 * self.delta_limit / (np.dot(np.transpose(x_k), fim).dot(x_k))) * x_k

                divergence = delta_theta_k.transpose().dot(fim).dot(delta_theta_k)
                while divergence >= self.delta_limit:
                    delta_theta_k *= 0.9
                    divergence = delta_theta_k.transpose().dot(fim).dot(delta_theta_k)
                delta_weight = delta_theta_k.reshape(self.action_space_size, -1)
                new_weight = self.weights_policy.clone().detach().numpy() + delta_weight
                self.weights_policy = torch.tensor(new_weight, requires_grad=True, dtype=torch.float32)
            elif self.method == 'NPG':
                delta_weight = np.linalg.inv(fim).dot(eta_summation)
                delta_weight /= (10*np.max(delta_weight))
                new_weight = self.weights_policy.clone().detach().numpy() + delta_weight.reshape(self.action_space_size, -1)
                self.weights_policy = torch.tensor(new_weight, requires_grad=True, dtype=torch.float32)

            # print log
            if epoch_i % 100 == 0:
                writer.add_scalar('reward', log_reward / (self.trajectory_num_per_update * 100.),
                                  self.trajectory_num_per_update * epoch_i)
                print('reward: ' + str(log_reward / (self.trajectory_num_per_update * 100.)))
                print('weights state value:')
                print(self.weights_state_value)
                print('weights of policy:')
                print(self.weights_policy)
                print('gradient estimate ')
                print(np.transpose(gradient_estimate))
                print('fisher information matrix:')
                print(fim)
                print('delta weight:')
                print(delta_weight)
                log_reward = 0
            # -----------------------------------


if __name__ == '__main__':
    env = tetris.Tetris(TETRIS_WIDTH, TETRIS_HEIGHT)
    trpo_agent = TRPO_Agent(env, delta_limit_=0.01, input_size_=TETRIS_WIDTH * TETRIS_HEIGHT + 5,method_='NPG')
    trpo_agent.optimization(100000)
