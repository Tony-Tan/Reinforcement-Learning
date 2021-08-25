import collections

import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env, epsilon=0.4):
        self.env = env
        self.epsilon = epsilon
        self.value_state_action = collections.defaultdict(float)
        self.value_state_action_visit_times = collections.defaultdict(lambda: 1.0)
        self.target_policies = collections.defaultdict(constant_factory(env.action_space.n))
        self.behavior_policies = collections.defaultdict(constant_factory(env.action_space.n))

    def select_action(self, state):
        probability_distribution = self.behavior_policies[state]
        action = np.random.choice(env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def play_1_episode(self):
        episode = []
        state = self.env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = self.env.step(action)
            episode.append([state, action, reward])
            if is_done:
                break
            state = new_state
        return episode

    def first_visit_mc(self, alpha=0.0, gamma=0.9, method='on-policy', only_evaluation=False):
        for _ in range(100):
            episode = self.play_1_episode()
            g = 0.0
            w = 1
            for i in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[i]
                g = gamma * g + reward
                is_first = True
                for j in range(i):
                    if episode[i][0] == episode[j][0] and episode[i][1] == episode[j][1]:
                        is_first = False
                        break
                if not is_first:
                    continue

                if method == 'on-policy':

                    old_value = self.value_state_action[(state, action)]
                    if alpha == 0:
                        average_times = self.value_state_action_visit_times[(state, action)]
                        self.value_state_action[(state, action)] = old_value + 1. / (average_times + 1) * (
                                    g - old_value)
                        self.value_state_action_visit_times[(state, action)] += 1
                    else:
                        self.value_state_action[(state, action)] = old_value + alpha * (
                                g - old_value)
                elif method == 'off-policy':
                    # incremental implement introduced in the book 'reinforcement learning: an introduction' 2ed
                    # section 5.6 C(s,a)
                    self.value_state_action_visit_times[(state, action)] += w
                    c_s_a = self.value_state_action_visit_times[(state, action)]
                    old_value = self.value_state_action[(state, action)]
                    self.value_state_action[(state, action)] = old_value + w / c_s_a * (g - old_value)
                    w *= 1. / self.behavior_policies[state][action]

                # update policy epsilon greedy
                if not only_evaluation:
                    value_list = [self.value_state_action[(state, action_iter)]
                                  for action_iter in range(self.env.action_space.n)]
                    action = np.argmax(value_list)

                    if method == 'on-policy':
                        for action_iter in range(self.env.action_space.n):
                            if action == action_iter:
                                self.target_policies[state][action_iter] = \
                                    1 - self.epsilon + self.epsilon / self.env.action_space.n
                                self.behavior_policies[state][action_iter] = \
                                    1 - self.epsilon + self.epsilon / self.env.action_space.n
                            else:
                                self.behavior_policies[state][action_iter] = \
                                    self.epsilon / self.env.action_space.n
                                self.target_policies[state][action_iter] = \
                                    self.epsilon / self.env.action_space.n
                    else:
                        for action_iter in range(self.env.action_space.n):
                            if action == action_iter:
                                self.target_policies[state][action_iter] = 1
                            else:
                                self.target_policies[state][action_iter] = 0
                        if self.target_policies[state][action] != 1:
                            break


if __name__ == '__main__':
    env = gym.make("Blackjack-v0")
    agent = Agent(env)
    """
    # on-policy
    agent.first_visit_mc(method='on-policy')
    state_action_without_ace = np.zeros([21, 10])
    state_action_with_ace = np.zeros([21, 10])
    state_action_0_1_with_ace = np.zeros([21, 10])
    state_action_0_1_without_ace = np.zeros([21, 10])
    for player in range(21):
        for dealer in range(10):
            state_action_with_ace[player][dealer] = np.argmax(agent.target_policies[(player + 1, dealer + 1, True)])
            state_action_without_ace[player][dealer] = np.argmax(agent.target_policies[(player + 1, dealer + 1, False)])
            state_action_0_1_with_ace[player][dealer] = \
                agent.value_state_action[((player + 1, dealer + 1, True), 0)] - \
                agent.value_state_action[((player + 1, dealer + 1, True), 1)]
            state_action_0_1_without_ace[player][dealer] = \
                agent.value_state_action[((player + 1, dealer + 1, False), 0)] - \
                agent.value_state_action[((player + 1, dealer + 1, False), 1)]
    Y = np.arange(0, 21, 1)
    X = np.arange(0, 10, 1)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(1, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, state_action_0_1_with_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("value$_{stick}$-value$_{hit}$")
        ax.view_init(30, angle)
        filename = "./mc_data/on-policy/value_stick-hit_with_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")

    fig = plt.figure(2, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, state_action_0_1_without_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("value$_{stick}$-value$_{hit}$")
        ax.view_init(30, angle)
        filename = "./mc_data/on-policy/value_stick-hit_without_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")

    fig = plt.figure(3, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, state_action_with_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("action: 0-stick 1-hit")
        ax.view_init(30, angle)
        filename = "./mc_data/on-policy/action_with_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")

    fig = plt.figure(4, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, state_action_without_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("action: 0-stick 1-hit")
        ax.view_init(30, angle)
        filename = "./mc_data/on-policy/action_without_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")
    """
    # off policy first visit MC to solve Blackjack
    agent.first_visit_mc(method='off-policy')
    state_action_without_ace = np.zeros([21, 10])
    state_action_with_ace = np.zeros([21, 10])
    state_action_0_1_with_ace = np.zeros([21, 10])
    state_action_0_1_without_ace = np.zeros([21, 10])
    for player in range(21):
        for dealer in range(10):
            state_action_with_ace[player][dealer] = np.argmax(agent.target_policies[(player + 1, dealer + 1, True)])
            state_action_without_ace[player][dealer] = np.argmax(agent.target_policies[(player + 1, dealer + 1, False)])
            state_action_0_1_with_ace[player][dealer] = \
                agent.value_state_action[((player + 1, dealer + 1, True), 0)] - \
                agent.value_state_action[((player + 1, dealer + 1, True), 1)]
            state_action_0_1_without_ace[player][dealer] = \
                agent.value_state_action[((player + 1, dealer + 1, False), 0)] - \
                agent.value_state_action[((player + 1, dealer + 1, False), 1)]
    Y = np.arange(0, 21, 1)
    X = np.arange(0, 10, 1)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(1, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, state_action_0_1_with_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("value$_{stick}$-value$_{hit}$")
        ax.view_init(30, angle)
        filename = "./mc_data/off-policy/value_stick-hit_with_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")

    fig = plt.figure(2, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, state_action_0_1_without_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("value$_{stick}$-value$_{hit}$")
        ax.view_init(30, angle)
        filename = "./mc_data/off-policy/value_stick-hit_without_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")

    fig = plt.figure(3, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, state_action_with_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("action: 0-stick 1-hit")
        ax.view_init(30, angle)
        filename = "./mc_data/off-policy/action_with_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")

    fig = plt.figure(4, figsize=(12, 9))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, state_action_without_ace, cmap=cm.hsv)
    for angle in range(120, 270, 2):
        ax.set_zlabel("action: 0-stick 1-hit")
        ax.view_init(30, angle)
        filename = "./mc_data/off-policy/action_without_ace/" + str(angle) + ".png"
        plt.savefig(filename)
        print("Save " + filename + " finish")
