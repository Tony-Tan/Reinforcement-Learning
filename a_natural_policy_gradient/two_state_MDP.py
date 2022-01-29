import numpy as np
import random


class Policy:
    def __init__(self):
        self.weights = np.array([np.log(9)+np.log(4), -np.log(4)])
        self.trans_matrix = np.zeros((2, 2))
        # state i: 0, j:1
        # action a_1: -1, a_2: 1
        self.action_space = [-1, 1]
        self.state_space = [0, 1]
        #       action     -1        1
        # state
        #   0              1         0
        #   1              2         0
        self.reward = np.array([[1, 0], [2, 0]])

    def sigmoid(self, x, a):
        return 1.0 / (1.0 + np.exp(-(self.weights[0] * x + self.weights[1]) * a))

    def stationary_distribution(self):
        # state i: 0, j:1
        # action a_1: -1, a_2: 1
        self.trans_matrix[0][0] = self.sigmoid(self.state_space[0], self.action_space[0])
        self.trans_matrix[0][1] = self.sigmoid(self.state_space[0], self.action_space[1])
        self.trans_matrix[1][0] = self.sigmoid(self.state_space[1], self.action_space[1])
        self.trans_matrix[1][1] = self.sigmoid(self.state_space[1], self.action_space[0])
        p_j = (1 - self.trans_matrix[0][0])/(self.trans_matrix[1][0] - self.trans_matrix[0][0] + 1)
        return np.array([1-p_j, p_j])

    def derivative(self, x, a):
        forward_res = self.sigmoid(x, a)
        partial_w_0 = (forward_res-1) * forward_res * x * a
        partial_w_1 = (forward_res-1) * forward_res * a
        return np.array([[partial_w_0], [partial_w_1]])

    def Fisher_matrix(self):
        f_matrix = np.zeros((2, 2))
        stat_distri = self.stationary_distribution()
        for s_i in range(len(self.state_space)):
            f_matrix_s = np.zeros((2, 2))
            for a_i in range(len(self.action_space)):
                x = self.state_space[s_i]
                a = self.action_space[a_i]
                p_a = self.sigmoid(x, a)
                log_p_w = np.array([[(1-p_a)*x*a, (1-p_a)*a]])
                log_p_w_t = log_p_w.transpose()
                f_matrix_s += log_p_w_t.dot(log_p_w)*p_a
            f_matrix += stat_distri[s_i]*f_matrix_s
        return f_matrix


def policy_gradient_gradient_update(alpha):
    policy = Policy()
    for i in range(1000000):
        delta_eta = 0
        stationary_distri = policy.stationary_distribution()

        for s_i in range(len(policy.state_space)):
            p_s = stationary_distri[s_i]
            for a_i in range(len(policy.action_space)):
                policy_derivative = policy.derivative(policy.state_space[s_i], policy.action_space[a_i])
                delta_eta += p_s * policy_derivative * policy.reward[s_i][a_i]
        # F = policy.Fisher_matrix()
        # delta_w = np.linalg.inv(F).dot(delta_eta)
        delta_w = delta_eta
        policy.weights += alpha*delta_w.transpose()[0]
        if i%1000 == 0:
            print('-----------------------------------')
            print('weight:'+str(policy.weights))
            print('stationary distri'+str(stationary_distri))
            print(policy.trans_matrix)


if __name__ == '__main__':
    policy_gradient_gradient_update(0.01)
