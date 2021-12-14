# A test for likelihood ration methods to minimize the object E[r(X)]
# where r(x) =
# r.v. X has a distribution of the form exp(-(x-theta_0)^2/theta_1)

import numpy as np
import matplotlib.pyplot as plt
import copy

N = 1000


class GaussianPolicy:
    def __init__(self):
        self.theta = np.ones(2)

    def generator(self):
        return np.random.normal(self.theta[0], self.theta[1])

    def __call__(self, x):
        return (1 / np.sqrt(2 * np.pi * self.theta[1] ** 2)) * np.exp(
            -(x - self.theta[0]) ** 2 / (2 * self.theta[1] ** 2))

    def derivative(self, x):
        a = 1 / (self.theta[1] * np.sqrt(2 * np.pi))
        nabula_theta = np.zeros(2)
        gp_value = a * np.exp(-(x - self.theta[0]) ** 2 / (2 * self.theta[1] ** 2))
        nabula_theta[0] = (x - self.theta[0]) * gp_value / (self.theta[1] ** 2)
        nabula_theta[1] = gp_value * (self.theta[0] ** 2 - 2 *
                                      self.theta[0] * x - self.theta[1] ** 2 + x ** 2) / (self.theta[1] ** 3)
        return nabula_theta

    def update_theta(self, nabula_theta, step_size):
        self.theta += step_size * nabula_theta
        if self.theta[1] < 0:
            self.theta[1] = -self.theta[1]


def reward(x):
    # theta_0 = 2
    # theta_1 = 3
    return (6. / (3 * np.sqrt(2 * np.pi))) * np.exp(-(x - 2) ** 2 / 18)


def likelihood_ratio(repeat_time):
    log_theta = []
    log_reward = []
    gp = GaussianPolicy()
    learning_rate = 1
    learning_rate_decay = 0.95
    for i in range(1, repeat_time):
        nabula_q = np.zeros(2)
        reward_i = 0
        for _ in range(N):
            x = gp.generator()
            r = reward(x)
            reward_i += r
            nabula_q += r * gp.derivative(x) / gp(x)
        nabula_q /= N
        reward_i /= N
        gp.update_theta(nabula_q, learning_rate)
        learning_rate *= learning_rate_decay
        log_reward.append(reward_i)
        log_theta.append(copy.deepcopy(gp.theta))
    plt.figure(0)
    log_theta = np.array(log_theta)
    plt.plot(log_theta[:, 0], label='$\mu$')
    plt.plot(log_theta[:, 1], label='$\delta$')
    plt.legend()
    plt.figure(1)
    plt.plot(log_reward, label='reward')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    likelihood_ratio(500)
    print('finish')
