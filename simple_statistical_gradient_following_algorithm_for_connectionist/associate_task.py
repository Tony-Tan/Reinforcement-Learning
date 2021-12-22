from k_arm_bandit import KArmedBandit
import numpy as np
import matplotlib.pyplot as plt


SQRT_PI = np.sqrt(np.pi)


def squashing_function(input):
    return 1./(1.+np.exp(-input))


class GaussianUnit:
    def __init__(self, input_size_, action_space_):
        self.weights_mean = np.zeros(input_size_)
        # self.weights_std = np.zeros(input_size_)
        self.mean = 0
        self.std = 1
        self.action_space = np.array(action_space_)
        pass

    def __call__(self, input):
        self.mean = squashing_function(self.weights_mean.dot(input))*10
        # self.std = squashing_function(self.weights_std.dot(input))*10 + 0.01
        action_prob = 1/(2*SQRT_PI*self.std)*np.exp(-0.5*((self.action_space-self.mean)/self.std)**2)+0.01
        action_prob /= np.sum(action_prob)

        return np.random.choice(self.action_space, 1, p=action_prob)[0]

    def characteristic_eligibility(self, x, y):
        partial_mu = (y-self.mean)/(self.std**2)
        # partial_std = ((y-self.mean)**2-self.std**2)/(self.std**3)
        nabla_weights_mean = partial_mu * x
        # nabla_weights_std = partial_std * x
        return nabla_weights_mean #, nabla_weights_std


def reinforce_algorithm(repeat_times=10000):
    # 10 arm bandits
    K = 10
    reward_log = []
    env_mean = np.random.normal(.0, 1.0, K)

    env = KArmedBandit(env_mean, np.ones(K))
    gu = GaussianUnit(K, range(K))
    state, reward, is_done, _ = env.step(0)
    alpha_mean = 0.01
    # alpha_std = 0.01
    # base_line_std = 10
    base_line_mean = 0

    for repeat_i in range(repeat_times):
        x = np.array(state)
        y = gu(x)
        state, reward, is_done, _ = env.step(y)
        r = reward
        # nabla_mean, nabla_std = gu.characteristic_eligibility(x, y)
        nabla_mean = gu.characteristic_eligibility(x, y)
        # gu.weights_std += (alpha_std * (r-base_line_std)*nabla_std)
        gu.weights_mean += (alpha_mean * (r-base_line_mean)*nabla_mean)
        if repeat_i % 100 == 0:
            reward_log.append(r)
    print(gu.mean)
    print(gu.weights_mean)
    print(env_mean)
    plt.plot(reward_log)
    plt.show()


if __name__ == '__main__':
    reinforce_algorithm()





