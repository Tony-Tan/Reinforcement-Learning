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


class LinearStateValue:
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
        self.state_value = LinearStateValue()

    def select_action(self, state):
        probability_distribution = []
        for action_iter in self.env.action_space:
            probability_distribution.append(self.policy(state, action_iter))
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def play(self, number_of_episodes, alpha_theta, alpha_w, gamma):
        left_policy_prob = []
        for eps_iter in range(number_of_episodes):
            reward_sum = 0
            state = self.env.reset()
            value_i = 1.
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                if not is_done:
                    delta = reward + gamma * self.state_value(new_state) - self.state_value(state)
                else:
                    delta = reward - self.state_value(state)
                delta_ln_theta = self.policy.derivative_ln(state, action)
                delta_state_value = self.state_value.derivative(state)
                self.state_value.weight += alpha_w * delta * delta_state_value
                self.policy.weight += alpha_theta * value_i * delta * delta_ln_theta
                value_i *= gamma
                reward_sum += reward
                if is_done:
                    break
                state = new_state
            if eps_iter % 100 == 0:
                np.set_printoptions(precision=11)
                print(eps_iter, self.policy(0, 0), self.policy(0, 1), self.state_value.weight, reward_sum)
                left_policy_prob.append(self.state_value.weight[1])
        return np.array(left_policy_prob)


if __name__ == '__main__':
    # for i in range(0, 1):
    episode_len = 50000
    repeat_time = 1
    steps = np.zeros(episode_len)

    for i in range(repeat_time):
        print('repeat time ' + str(i))
        env = ShortCorridor()
        agent = Agent(env)
        step = agent.play(episode_len, 1e-3, 1e-2, 0.9)
        # steps += step
        plt.plot(step, alpha=0.7, label='$\\alpha_{\\theta}=1e-3,\\alpha_w=1e-2$')

        agent = Agent(env)
        step = agent.play(episode_len, 1e-3, 1e-4, 0.9)
        # steps += step
        plt.plot(step, alpha=0.7, label='$\\alpha_{\\theta}=1e-3,\\alpha_w=1e-4$')

        agent = Agent(env)
        step = agent.play(episode_len, 1e-2, 1e-4, 0.9)
        # steps += step
        plt.plot(step, alpha=0.7, label='$\\alpha_{\\theta}=1e-2,\\alpha_w=1e-4$')

        plt.legend()
        plt.show()
    # plt.plot(steps / repeat_time, alpha=0.7, c='r', label='$\\alpha_{\\theta}=1e-3,\\alpha_w=1e-3$')
    # plt.show()
