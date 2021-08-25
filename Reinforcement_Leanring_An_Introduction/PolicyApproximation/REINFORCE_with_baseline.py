import matplotlib.pyplot as plt
import numpy as np
from environment.corridor_gridworld import ShortCorridor


class LogisticalFunc:
    def __call__(self, w):
        return np.exp(w) / (1. + np.exp(w)) * 0.9 + 0.05

    def derivative(self, w):
        return np.exp(w) / ((1. + np.exp(w)) ** 2) * 0.9


class MyPolicy:
    """
    p = (1-l(w))^x l(w)^{1-x}
    """

    def __init__(self):
        self.weight = 2.
        self.l = LogisticalFunc()

    def __call__(self, state, action):
        x = action
        return np.power((1 - self.l(self.weight)), x) * np.power(self.l(self.weight), 1 - x)

    def derivative_ln(self, state, action):
        x = action
        delta_p = -x * np.power((1 - self.l(self.weight)), x - 1) * self.l.derivative(self.weight) * np.power(
            self.l(self.weight), 1 - x) + np.power((1 - self.l(self.weight)), x) * (1 - x) * self.l.derivative(
            self.weight) * np.power(self.l(self.weight), - x)
        return delta_p / (self.__call__(state, action))


class Linear_State_Value:
    def __init__(self):
        self.weight = np.zeros(2)

    def __call__(self, x):
        return x * self.weight[0] + self.weight[1]

    def derivative(self, x):
        return np.array([x, 1.])


class Agent:
    def __init__(self, env):
        self.env = env
        self.policy = MyPolicy()
        self.state_value = Linear_State_Value()

    def select_action(self, state):
        probability_distribution = []
        exp_sum = 0
        for action_iter in self.env.action_space:
            probability_distribution.append(self.policy(state, action_iter) * 5.)
            exp_sum += np.exp(self.policy(state, action_iter) * 5.)
        for i in range(len(probability_distribution)):
            probability_distribution[i] = np.exp(probability_distribution[i]) / exp_sum
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def play(self, number_of_episodes, alpha_theta, alpha_w, gamma):
        reward_per_episode = []
        for eps_iter in range(number_of_episodes):
            reward_sum = 0
            episode = []
            state = self.env.reset()
            gamma_ = 1.
            total_g = 0
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                episode.append([state, action, reward])
                total_g += gamma_ * reward
                gamma_ *= gamma
                reward_sum += reward
                if is_done:
                    break
                state = new_state
            reward_per_episode.append(reward_sum)
            gamma_t = 1.
            for epd_i in range(len(episode) - 1):
                state, action, r = episode[epd_i]
                delta = total_g - self.state_value(state)
                delta_ln_theta = self.policy.derivative_ln(state, action)
                delta_state_value = self.state_value.derivative(state)
                self.state_value.weight += alpha_w * delta * delta_state_value
                self.policy.weight += alpha_theta * gamma_t * delta * delta_ln_theta
                gamma_t *= gamma
                total_g -= r
                total_g /= gamma
            # np.set_printoptions(precision=11)
            # print(eps_iter, self.policy(0, 0), self.state_value.weight)
        return np.array(reward_per_episode)


if __name__ == '__main__':

    # for i in range(0, 1):

    episode_len = 1000
    repeat_time = 10
    steps = np.zeros(episode_len)

    for i in range(repeat_time):
        print('repeat time ' + str(i))
        env = ShortCorridor()
        agent = Agent(env)
        steps += agent.play(episode_len, 2e-3, 2e-3, 0.99)
    plt.plot(steps / repeat_time, alpha=0.7, label='$\\alpha_{\\theta}=2e-3,\\alpha_w=2e-3$')

    steps = np.zeros(episode_len)
    for i in range(repeat_time):
        print('repeat time ' + str(i))
        env = ShortCorridor()
        agent = Agent(env)
        steps += agent.play(episode_len, 2e-4, 0, 0.99)
    plt.plot(steps / repeat_time, alpha=0.7, label='$\\alpha_{\\theta}=2e-3,\\alpha_w=0$')
    plt.legend()
    plt.show()
