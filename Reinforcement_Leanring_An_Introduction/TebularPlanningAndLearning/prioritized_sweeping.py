import collections
import random

import matplotlib.pyplot as plt
import numpy as np
from environment.gride_world import GridWorld


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env, n=5, epsilon=0.4, initial_value=0.0, theta=0.1):
        self.env = env
        self.n = n
        self.epsilon = epsilon
        self.value_state_action = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))
        self.model = collections.defaultdict(lambda: {})
        self.predicted_to_the_state = collections.defaultdict(lambda: {})
        self.PQueue = collections.deque()
        self.total_step_num = 0
        self.theta = theta

    def select_action(self, state):
        probability_distribution = self.policies[state]
        action = np.random.choice(self.env.action_space.n, 1, p=probability_distribution)
        return action[0]

    def update_policy(self, state, epsilon):
        # update policy
        value_of_action_list = []
        possible_action_num = self.env.action_space.n
        for action_iter in range(possible_action_num):
            value_of_action_list.append(self.value_state_action[(state, action_iter)])
        value_of_action_list = np.array(value_of_action_list)
        optimal_action = \
            np.random.choice(np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        for action_iter in range(self.env.action_space.n):
            if action_iter == optimal_action:
                self.policies[state][action_iter] = 1 - epsilon + epsilon / possible_action_num
            else:
                self.policies[state][action_iter] = epsilon / possible_action_num

    def inset_queue(self, prob_of_insertion, state_action_pair):
        if prob_of_insertion < self.theta:
            return False
        if random.random() < prob_of_insertion:
            self.PQueue.append(state_action_pair)
            return True

    def prioritized_sweeping(self, number_of_episodes, alpha=0.1, gamma=1., epsilon=0.1):
        steps_list = []
        for epi_iter in range(number_of_episodes):
            state = self.env.reset()
            step_num = 0
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                # q learning for probability basing on which we decide whether to insert the state-action pair
                q_state_next = []
                for action_iter in range(self.env.action_space.n):
                    q_state_next.append(self.value_state_action[(new_state, action_iter)])
                q_state_next = max(q_state_next)
                q_state_current = self.value_state_action[(state, action)]
                probability_of_insertion = np.abs(reward + gamma * q_state_next - q_state_current)
                self.inset_queue(probability_of_insertion, [state, action])
                # model learning
                self.model[state][action] = [reward, new_state]
                self.predicted_to_the_state[new_state][(state, action)] = reward
                # planning
                for n_iter in range(self.n):
                    if not len(self.PQueue) > 0:
                        break
                    state_selected, action_selected = self.PQueue.popleft()
                    reward_in_model, new_state_in_model = self.model[state_selected][action_selected]
                    q_state_next_in_model = []
                    for action_iter in range(self.env.action_space.n):
                        q_state_next_in_model.append(self.value_state_action[(new_state_in_model, action_iter)])
                    q_state_next_in_model = max(q_state_next_in_model)
                    q_state_current_in_model = self.value_state_action[(state_selected, action_selected)]
                    self.value_state_action[(state_selected, action_selected)] = \
                        q_state_current_in_model + alpha * (reward_in_model + gamma * q_state_next_in_model -
                                                            q_state_current_in_model)
                    # loop for all \bar{s}, \bar{a} predicted to lead to S
                    for state_action_iter in self.predicted_to_the_state[state_selected].keys():
                        pre_state, pre_action = state_action_iter
                        pre_reward = self.predicted_to_the_state[state_selected][(pre_state, pre_action)]
                        # q learning insertion in planning
                        q_state_next = []
                        for action_iter in range(self.env.action_space.n):
                            q_state_next.append(self.value_state_action[(state_selected, action_iter)])
                        q_state_next = max(q_state_next)
                        q_state_current = self.value_state_action[(pre_state, pre_action)]
                        probability_of_insertion = np.abs(pre_reward + gamma * q_state_next - q_state_current)
                        self.inset_queue(probability_of_insertion, [pre_state, pre_action])
                    # update policy in planning
                    self.update_policy(state_selected, epsilon)
                if is_done:
                    break
                self.total_step_num += 1
                step_num += 1
                state = new_state
            steps_list.append(self.total_step_num)
        return steps_list


if __name__ == '__main__':
    # env = GridWorld(6, [25, 26, 27, 28, 29], start_position=31, end_position_list=[5])
    # steps_matrix = np.zeros((3, 50))
    # agent_n = [0, 5, 50]
    # repeat_n_times = 10
    # for j in range(repeat_n_times):
    #     for i in range(3):
    #         agent = Agent(env, n=agent_n[i], kappa=0)
    #         steps = agent.dyna_q_plus(50, alpha=0.1, gamma=0.95,epsilon=.3)
    #         steps_matrix[i] += steps
    # steps_matrix /= 50.
    # i = 0
    # for steps_array in steps_matrix:
    #     plt.plot(steps_array,label=agent_n[i])
    #     i += 1
    # plt.legend()
    # plt.show()

    env = GridWorld(6, [25, 26, 27, 28, 29], start_position=31, end_position_list=[5])
    agent = Agent(env, n=5, theta=0.01)
    steps = 0
    pw_step_list = agent.prioritized_sweeping(400, alpha=0.1, gamma=0.95, epsilon=.3)
    pw_step_list = np.array(pw_step_list)
    plt.plot(pw_step_list, np.arange(1, 401), label='Prioritized Sweeping')
    agent.env.plot_grid_world(agent.policies)
    plt.legend()
    plt.show()
