import numpy as np
import random


class Policy:
    def __init__(self):
        self.weights = np.array([-np.log(4), -np.log(9)])
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

    def q_value_matrix(self, gamma):
        # pi(a|x) = pi_ax
        pi_00 = self.sigmoid(self.state_space[0], self.action_space[0])
        pi_01 = 1 - pi_00
        pi_10 = self.sigmoid(self.state_space[0], self.action_space[1])
        pi_11 = 1 - pi_10
        v_0 = (2 * pi_01 - (1 - gamma * pi_01) * (
                    gamma * pi_00 * pi_11 - 2 * (1 - gamma * pi_00) * pi_01 / ((1 - gamma * pi_00)
                    * (1 - gamma * pi_01) + gamma * gamma * pi_10 * pi_11))) / (-gamma * pi_11)
        v_1 = (gamma * pi_00 * pi_11 + 2 * (1 - gamma * pi_00) * pi_01) / ((1 - gamma * pi_00) * (1 - gamma * pi_00) *
                    (1 - gamma * pi_01) + gamma * gamma * pi_10 * pi_11)
        return np.array([[1 + gamma * v_0, gamma * v_1], [2 + gamma * v_1, gamma * v_0]])

    def sigmoid(self, x, a):
        return 1.0 / (1.0 + np.exp(-(self.weights[0]*a)))*(1-x) + 1.0 / (1.0 + np.exp(-(self.weights[1]*a)))*x

    def stationary_distribution(self):
        # state i: 0, j:1
        # action a_1: -1, a_2: 1
        self.trans_matrix[0][0] = self.sigmoid(self.state_space[0], self.action_space[0])
        self.trans_matrix[0][1] = self.sigmoid(self.state_space[0], self.action_space[1])
        self.trans_matrix[1][0] = self.sigmoid(self.state_space[1], self.action_space[1])
        self.trans_matrix[1][1] = self.sigmoid(self.state_space[1], self.action_space[0])
        p_j = (1 - self.trans_matrix[0][0]) / (self.trans_matrix[1][0] - self.trans_matrix[0][0] + 1)
        return np.array([1 - p_j, p_j])

    def derivative(self, x, a):
        if x ==0:
            forward_res = 1.0 / (1.0 + np.exp(-(self.weights[0] * a)))
            partial_w_0 = (1-forward_res) * forward_res * a
            partial_w_1 = 0
            return np.array([[partial_w_0], [partial_w_1]])
        elif x==1:
            forward_res = 1.0 / (1.0 + np.exp(-(self.weights[1] * a)))
            partial_w_0 = 0
            partial_w_1 = (1 - forward_res) * forward_res * a
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
                log_p_w = np.array([[(1 - p_a) * x * a, (1 - p_a) * a]])
                log_p_w_t = log_p_w.transpose()
                f_matrix_s += log_p_w_t.dot(log_p_w) * p_a
            f_matrix += stat_distri[s_i] * f_matrix_s
        return f_matrix + 1e-3 * np.eye(2)


def policy_gradient_gradient_update(alpha):
    policy = Policy()
    gamma = 0.8
    for i in range(20000000):
        delta_eta = 0
        stationary_distri = policy.stationary_distribution()
        q_matrix = policy.q_value_matrix(gamma)
        for s_i in range(len(policy.state_space)):
            p_s = stationary_distri[s_i]
            # p_s = 0.5
            for a_i in range(len(policy.action_space)):
                policy_derivative = policy.derivative(policy.state_space[s_i], policy.action_space[a_i])
                delta_eta += p_s * policy_derivative * q_matrix[s_i][a_i]  #policy.reward[s_i][a_i]    # #
        # F = policy.Fisher_matrix()
        # delta_w = np.linalg.inv(F).dot(delta_eta)
        delta_w = delta_eta
        policy.weights += alpha * delta_w.transpose()[0]
        if i % 100000 == 0:
            print('---------------%dth iteration--------------------'%i)
            print('weight:' + str(policy.weights))
            print('stationary distri' + str(stationary_distri))
            print(policy.trans_matrix)


if __name__ == '__main__':
    policy_gradient_gradient_update(0.01)
