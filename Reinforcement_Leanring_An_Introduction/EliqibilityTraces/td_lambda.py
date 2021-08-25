import collections

import matplotlib.pyplot as plt
import numpy as np
from environment.random_walk_19_states import RandomWalk


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env):
        self.env = env
        self.policies = collections.defaultdict(constant_factory(2))
        self.value_of_state = collections.defaultdict(lambda: 0.0)

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def estimating(self, iteration_times, lambda_coe=1., alpha=0.1, gamma=0.9, epsilon=0.3):
        for iter_time in range(iteration_times):
            eligibility_trace = collections.defaultdict(lambda: 0.0)
            current_state = self.env.reset()
            current_action = self.select_action(current_state)
            next_state, reward, is_done, _ = self.env.step(current_action)
            while True:
                for k in self.value_of_state.keys():
                    eligibility_trace[k] = gamma * lambda_coe * eligibility_trace[k]
                eligibility_trace[current_state] += 1.
                if not is_done:
                    delta_value = reward + gamma * self.value_of_state[next_state] - \
                                  self.value_of_state[current_state]
                else:
                    delta_value = reward - self.value_of_state[current_state]
                for k in self.value_of_state.keys():
                    self.value_of_state[k] += alpha * delta_value * eligibility_trace[k]
                # update policy
                # value_of_action_list = []
                # for action_iter in range(self.env.action_space.n):
                #     value_of_action_list.append(self.value_of_state[current_state])
                # value_of_action_list = np.array(value_of_action_list)
                # optimal_action = np.random.choice(
                #     np.flatnonzero(value_of_action_list == value_of_action_list.max()))
                # for action_iter in range(self.env.action_space.n):
                #     if action_iter == optimal_action:
                #         self.policies[current_state][
                #             action_iter] = 1 - epsilon + epsilon / self.env.action_space.n
                #     else:
                #         self.policies[current_state][action_iter] = epsilon / self.env.action_space.n

                if is_done:
                    break
                current_state = next_state
                current_action = self.select_action(current_state)
                next_state, reward, is_done, _ = self.env.step(current_action)


if __name__ == '__main__':
    env = RandomWalk(19)
    alpha_list = [i / 50. for i in range(1, 51)]
    lambda_list = [1., 0.99, 0.975, 0.95, 0.9, 0.8, 0.4, 0]
    ground_truth = [(-1. + i / 9.) for i in range(0, 19)]
    ground_truth = np.array(ground_truth)
    average_times = 200
    max_rms = 0.6
    for j in range(len(lambda_list)):
        mse = []
        for alpha_i in range(0, len(alpha_list)):

            mse_current = 0
            for aver in range(average_times):
                agent = Agent(env)
                agent.estimating(10, lambda_list[j], alpha_list[alpha_i], 0.99)
                value_of_state = []
                for i_state in range(0, env.state_space.n):
                    value_of_state.append(agent.value_of_state[i_state])
                value_of_state = np.array(value_of_state)
                mse_current += np.sum((value_of_state - ground_truth) *
                                      (value_of_state - ground_truth)) / 19.
            rms = np.sqrt(mse_current / float(average_times))
            if rms > max_rms:
                break
            print(alpha_i, alpha_list[alpha_i], np.sqrt(mse_current / float(average_times)))
            mse.append(rms)

        plt.plot(alpha_list[0:len(mse)], mse, label='$\\lambda = $' + str(lambda_list[j]))
    plt.legend()
    plt.show()

    # env = RandomWalk(19)
    # agent = Agent(env)
    # agent.estimating(10, 0.8, 0.4, 0.9)
    # value_of_state = []
    # for i_state in range(1, env.state_space.n - 1):
    #     value_of_state.append(agent.value_of_state[i_state])
    # plt.plot(value_of_state)
    # plt.show()
