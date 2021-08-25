import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.access_control_queuing_task import QueuingTask


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Tabular:
    def __init__(self, tabular_size):
        # tabular_size contents three dimensions
        # 1: state dimension 1
        # 2: state dimension 2
        # 3: action
        self.weights = np.zeros(tabular_size)

    def __call__(self, state_action_pair):
        return self.weights[state_action_pair[0][0]][state_action_pair[0][1]][state_action_pair[1]]

    def update_weight(self, delta_value, alpha, state_action_pair):
        self.weights[state_action_pair[0][0]][state_action_pair[0][1]][state_action_pair[1]] += alpha * delta_value * 1


class Agent:
    def __init__(self, environment):
        self.env = environment
        self.value_of_state_action = Tabular([environment.customer_kind_number,
                                              environment.sever_num + 1,
                                              self.env.action_space.n])
        self.policies = collections.defaultdict(constant_factory(self.env.action_space.n))
        self.average_reward = 0

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def running(self, max_step_num, alpha=0.01, beta=0.01, epsilon=0.1):
        state = self.env.reset()
        action = self.select_action(state)
        # get reward R and next state S'
        for i in range(max_step_num):
            next_state, reward, is_done, _ = self.env.step(action)
            next_action = self.select_action(next_state)
            delta = reward - self.average_reward + self.value_of_state_action([next_state, next_action]) \
                    - self.value_of_state_action([state, action])
            self.average_reward += beta * delta
            self.value_of_state_action.update_weight(delta, alpha, (state, action))

            # update policy
            value_of_action_list = []
            for action_iter in range(self.env.action_space.n):
                value_of_action_list.append(self.value_of_state_action((state, action_iter)))
            value_of_action_list = np.array(value_of_action_list)
            optimal_action = np.random.choice(
                np.flatnonzero(value_of_action_list == value_of_action_list.max()))
            for action_iter in range(self.env.action_space.n):
                if action_iter == optimal_action:
                    self.policies[state][
                        action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                else:
                    self.policies[state][action_iter] = epsilon / self.env.action_space.n
            # go to next step
            action = self.select_action(next_state)
            state = next_state


if __name__ == '__main__':
    rewards_list = [1, 2, 4, 8]
    rewards_distribution = [0.25, 0.25, 0.25, 0.25]
    env = QueuingTask(rewards_list, rewards_distribution)
    agent = Agent(env)
    agent.running(1000000)
    diff_value_matrix = []
    for i in range(4):
        diff_value_list = []
        for j in range(0, 11):
            print(agent.policies[(i, j)], end=' ')
            if agent.policies[(i, j)][0] > agent.policies[(i, j)][1]:
                diff_value_list.append(agent.value_of_state_action(([i, j], 0)))
            else:
                diff_value_list.append(agent.value_of_state_action(([i, j], 1)))
        diff_value_matrix.append(diff_value_list)
        print('\n')

    for i in range(len(diff_value_matrix)):
        plt.plot(diff_value_matrix[i], label='priority ' + str(rewards_list[i]))
    plt.legend()
    plt.show()
