import collections

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
    def __init__(self, environment, n):
        self.env = environment
        self.n = n
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
        n_queue = collections.deque()
        for i in range(max_step_num):
            next_state, reward, is_done, _ = self.env.step(action)
            n_queue.append([state, action, reward, next_state])
            if len(n_queue) < self.n + 1:
                action = self.select_action(next_state)
                state = next_state
                continue

            if len(n_queue) == self.n + 1:
                state_tao_n = n_queue[-1][3]
                action_tao_n = self.select_action(state_tao_n)
                delta_r = 0
                for rec_i in n_queue:
                    delta_r += (rec_i[2] - self.average_reward)
                state_2_update, action_2_update, reward, _ = n_queue.popleft()
                delta = delta_r + self.value_of_state_action([state_tao_n, action_tao_n]) \
                        - self.value_of_state_action([state_2_update, action_2_update])
                self.average_reward += beta * delta
                self.value_of_state_action.update_weight(delta, alpha, (state_2_update, action_2_update))
                # update policy
                value_of_action_list = []
                for action_iter in range(self.env.action_space.n):
                    value_of_action_list.append(self.value_of_state_action((state_2_update, action_iter)))
                value_of_action_list = np.array(value_of_action_list)
                optimal_action = np.random.choice(
                    np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                for action_iter in range(self.env.action_space.n):
                    if action_iter == optimal_action:
                        self.policies[state_2_update][
                            action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                    else:
                        self.policies[state_2_update][action_iter] = epsilon / self.env.action_space.n
                # go to next step
                next_state = state_tao_n
                action = self.select_action(next_state)
                state = next_state


if __name__ == '__main__':
    rewards_list = [1, 2, 4, 8]
    rewards_distribution = [0.25, 0.25, 0.25, 0.25]
    env = QueuingTask(rewards_list, rewards_distribution)
    agent = Agent(env, 4)
    agent.running(1000000)
    for i in range(4):
        for j in range(1, 11):
            # if agent.policies[(i, j)][0] > agent.policies[(i, j)][1]:
            #     print('Reject', end=' ')
            # else:
            #     print('Accept', end=' ')
            print(agent.policies[(i, j)], end=' ')
        print('\n')
