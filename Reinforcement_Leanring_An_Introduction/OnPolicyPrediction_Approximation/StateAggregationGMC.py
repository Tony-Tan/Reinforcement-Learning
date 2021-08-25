import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk_1000_states import RandomWalk1000


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class StateAggregation:
    def __init__(self, min_state, max_state, aggregation_size):
        self.min_state = min_state
        self.max_state = max_state
        self.aggregation_size = aggregation_size
        self.aggregation_num = int((max_state - min_state) / aggregation_size) + 1
        if (max_state - min_state) % aggregation_size == 0:
            self.aggregation_num -= 1
        self.weight = np.zeros(self.aggregation_num)

    def __call__(self, x):
        current_position = int(x / self.aggregation_size)
        return self.weight[current_position]

    def derivation(self, x):
        derivative = np.zeros(self.aggregation_num)
        current_position = int(x / self.aggregation_size)
        derivative[current_position] = 1.0
        return derivative


class Agent:
    def __init__(self, env, min_state, max_state, aggregation_size):
        self.env = env
        self.value_state = StateAggregation(min_state, max_state, aggregation_size)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def MC_app(self, number_of_episodes, learning_rate, state_num=1000, gamma=1):
        mu = np.zeros(state_num)
        for _ in range(number_of_episodes):
            episode = []
            state = self.env.reset()
            action = self.select_action(state)
            episode.append([0, state, action])
            while True:
                mu[state] += 1.0
                new_state, reward, is_done, _ = self.env.step(action)
                action = self.select_action(state)
                state = new_state
                episode.append([reward, state, action])
                if is_done:
                    break
            # update g base on g = gamma * g + R_{t+1}
            g = 0
            for i in range(len(episode) - 1, -1, -1):
                g = gamma * g + episode[i][0]
                # g += episode[i][0]
                episode[i][0] = g

            for i in range(0, len(episode)):
                s = episode[i][1]
                if s is None:
                    continue
                g = episode[i][0]
                # s /= 1000.
                delta_value = self.value_state.derivation(s)
                self.value_state.weight += learning_rate * (g - self.value_state(s)) * delta_value
        return mu


if __name__ == '__main__':
    env = RandomWalk1000()
    agent = Agent(env, 0, 1000, 100)
    mu = agent.MC_app(100000, 2e-5)
    mu = mu / np.sum(mu)
    x = np.arange(1, 999, 1.)
    y = np.arange(1, 999, 1.)
    # for i in range(1, x.size, 2):
    #     y[i-1] = agent.value_state(x[i-1] + 50)
    #     y[i] = agent.value_state(x[i] - 50)
    print(agent.value_state.weight)
    for i in range(x.size):
        y[i] = agent.value_state(x[i])
    plt.figure(0)
    plt.plot(x, y)
    plt.figure(1)
    plt.bar(range(len(mu)), mu, color='gray', width=1)
    plt.show()
