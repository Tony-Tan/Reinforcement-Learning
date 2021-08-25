import matplotlib.pyplot as plt
import numpy as np


class RLEnvironment:
    def __init__(self, n):
        self.gridword_size = n  # n x n grids
        self.dynamic = 1
        self.termination = [[0, 0], [n - 1, n - 1], [int(n / 2), int(n / 2)]]

    def get_reward(self, state):
        return -1

    def is_termination(self, state):
        if state in self.termination:
            return True


class Agent:
    def __init__(self, environment):
        self.degree_of_freedom = 4
        self.state = environment
        self.n = environment.gridword_size
        self.policy = np.ones([self.n, self.n, self.degree_of_freedom])
        self.value_function = np.zeros([self.n, self.n])
        # action: left, up, right, down
        self.action = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        # inital policy
        n = self.n
        for i in range(n):
            for j in range(n):
                if self.state.is_termination([i, j]):
                    self.policy[i][j] = 0
                else:
                    freedom = 0
                    if j - 1 >= 0:
                        freedom += 1
                        self.policy[i][j][0] = 1
                    if i - 1 >= 0:
                        freedom += 1
                        self.policy[i][j][1] = 1
                    if j + 1 < n:
                        freedom += 1
                        self.policy[i][j][2] = 1
                    if i + 1 < n:
                        freedom += 1
                        self.policy[i][j][3] = 1
                    self.policy[i][j] /= freedom

    def update_policy_greedy(self):
        self.policy = self.get_greedy_policy()

    def get_greedy_policy(self):
        n = self.n
        greedy_policy = np.ones([self.n, self.n, self.degree_of_freedom])
        for i in range(n):
            for j in range(n):
                if self.state.is_termination([i, j]):
                    continue
                value_of_successor_s = []
                for a_ in range(self.degree_of_freedom):
                    action = self.action[a_]
                    if 0 <= i + action[0] < n and 0 <= j + action[1] < n:
                        value_of_successor_s.append(self.value_function[i + action[0], j + action[1]])
                    else:
                        value_of_successor_s.append(-10000.)
                value_of_successor_s = np.array(value_of_successor_s)
                greedy_policy[i][j] = 0
                greedy_policy[i][j][np.argmax(value_of_successor_s)] = 1
        return greedy_policy

    def print_policy_and_value(self):
        n = self.n
        size_of_policy = len(self.policy[0][0])
        plt.figure(figsize=(8, 8))
        for i in range(n):
            for j in range(n):
                if self.state.is_termination([i, j]):
                    plt.text((j + 1.05) / (n + 1), 1 - (i + 0.95) / (n + 1), 'End',
                             size='medium')
                    continue
                current_policy = self.policy[i][j]
                for a_ in range(size_of_policy):
                    if current_policy[a_] != 0:
                        plt.arrow((j + 1) / (n + 1), 1 - (i + 1) / (n + 1), self.action[a_][1] / (4 * (n + 1)),
                                  -self.action[a_][0] / (4 * (n + 1)), head_width=0.01, head_length=0.01, fc='k',
                                  ec='k')
                        plt.text((j + 1.05) / (n + 1), 1 - (i + 0.95) / (n + 1),
                                 str(self.value_function[i][j].round(2)), size='medium')
        plt.show()

    def policy_evluation(self, gamma, threshold_of_termination, repeat_times=10000, method='in_place'):
        for epoch in range(repeat_times):
            print(self.value_function)
            self.print_policy_and_value()
            print('=========================================')
            value_delta = 0
            # recording new values
            value_function_new = np.zeros([self.n, self.n])
            # loop of all actions
            for i in range(self.n):
                for j in range(self.n):
                    if self.state.is_termination([i, j]):
                        continue
                    value_old = self.value_function[i][j]
                    temple_value = 0
                    # loop of policy
                    policy_of_s = self.policy[i][j]
                    for a_ in range(self.degree_of_freedom):
                        if policy_of_s[a_] == 0:
                            continue
                        action = self.action[a_]
                        # loop of next state and reward
                        if 0 <= action[0] + i < self.state.gridword_size and \
                                0 <= action[1] + j < self.state.gridword_size:
                            v_s_of_successor = self.value_function[action[0] + i][action[1] + j]
                            r = self.state.get_reward([action[0] + i, action[1] + j])
                        else:
                            continue
                        temple_value += policy_of_s[a_] * (r + gamma * v_s_of_successor)
                    if method == 'in_place':
                        self.value_function[i][j] = temple_value
                    else:
                        value_function_new[i][j] = temple_value
                    value_delta = np.max([value_delta, np.abs(value_old - temple_value)])
            if method != 'in_place':
                self.value_function = value_function_new

            if value_delta < threshold_of_termination:
                break
        return value_delta

    def policy_iteration(self, gamma, threshold_of_termination, method='in_place'):
        while True:
            value_delta = self.policy_evluation(gamma, threshold_of_termination, repeat_times=1, method=method)
            if value_delta < threshold_of_termination:
                return value_delta
            self.update_policy_greedy()


if __name__ == '__main__':
    env = RLEnvironment(10)
    agt = Agent(env)
    # agt.policy_evluation(1, 0.0001, repeat_times=100, method='not in place')
    agt.policy_iteration(0.9, 0.0001, method='not in place')
