import numpy as np
import random


class Policy:
    def __init__(self):
        self.weights = np.array([-np.log(4)-np.log(9), np.log(4)])
        self.action_distribute = 0

    def __call__(self, x):
        x = np.array([x, 1])
        prob_of_action1 = (np.exp(self.weights.dot(x)) / (1.0 + np.exp(self.weights.dot(x))))
        self.action_distribute = [prob_of_action1, 1. - prob_of_action1]
        return self.action_distribute

    def action_generator(self):
        return np.random.choice([0, 1], 1, p=self.action_distribute)

    def derivative(self, x):
        forward_res = self.__call__(x)[0]
        partial_w_1 = forward_res * (1 - forward_res) * self.weights[0]
        partial_w_2 = forward_res * (1 - forward_res)
        return np.array([partial_w_1, partial_w_2])

    def Fisher_matrix(self):
        F = 0
        trans_matrix = np.array([self.__call__(0), self.__call__(1)])
        p_state0 = trans_matrix[1][0] / (1 - trans_matrix[0][0] + trans_matrix[1][0])
        prob_s = [p_state0, 1. - p_state0]
        for x in [0, 1]:
            coff = (1 - self.__call__(x)[0])
            f_as = np.array([[(coff*x)**2, coff**2*x], [coff**2*x, coff**2]])
            F += f_as * prob_s[x]

        return F


def policy_gradient_gradient_update(alpha):
    policy = Policy()
    for i in range(1000):
        delta_eta = policy.derivative(0)*1 + policy.derivative(1)*2
        delta_w = np.linalg.inv(policy.Fisher_matrix()).dot(delta_eta)
        policy.weights += alpha*delta_w
        print(policy(0), policy(1))


if __name__ == '__main__':
    policy_gradient_gradient_update(0.0001)
