# Williams, Ronald J. “Simple Statistical Gradient-Following Algorithms for
# Connectionist Reinforcement Learning.” Machine Learning 8, no. 3 (1992): 229–56.
# episodic task
# example 13.1 short corridor with switched actions in reinforcement leanring: an introduction
# by Richard S. Sutton
import copy

import short_corridor_with_switched_actions as scwsa
import numpy as np
import matplotlib.pyplot as plt


def squashing_function(s):
    return 1. / (1. + np.exp(-s))


class Agent:
    def __init__(self):
        self.weight = np.random.random(1)[0]

    def policy(self, x):
        action_distribution = squashing_function(self.weight * x)

        action = np.random.choice([0, 1], p=[action_distribution, 1-action_distribution])
        return action

    def run(self, repeat_time):
        env = scwsa.ShortCorridorWithSwitchedActions()
        alpha = 1e-12
        baseline = 0
        reward_log = []
        for epoch_i in range(repeat_time):
            trace = []
            state = env.reset()
            is_done = False
            action = self.policy(state)
            reward_sum = 0
            while not is_done:
                next_state, reward, is_done, _ = env.step(action)
                trace.append([state, action, reward])
                reward_sum += reward
                state = next_state
                action = self.policy(state)
            reward_log.append(copy.deepcopy(reward_sum))
            for t_i in trace:
                sum_r = reward_sum - t_i[2]
                x = t_i[0]
                s = self.weight * x
                delta_weight = np.exp(-s) / ((1 + np.exp(-s)) ** 2) * x
                self.weight += alpha*(sum_r - baseline)*delta_weight
        plt.plot(reward_log)
        plt.show()


if __name__ == '__main__':
    agent = Agent()
    agent.run(10000)





