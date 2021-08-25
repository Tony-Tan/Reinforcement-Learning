import collections
import random

import matplotlib.pyplot as plt
import numpy as np
from environment.gride_world import GridWorld


def constant_factory(n):
    probability_list = np.ones(n)
    return lambda: probability_list / np.sum(probability_list)


class Agent:
    def __init__(self, env, n=5, epsilon=0.4, initial_value=0.0, kappa=0.00001):
        self.env = env
        self.epsilon = epsilon
        self.value_state_action = collections.defaultdict(lambda: initial_value)
        self.policies = collections.defaultdict(constant_factory(env.action_space.n))
        self.model = collections.defaultdict(lambda: {})
        self.model_state_action_list = []
        self.n = n
        self.total_step_num = 0
        self.kappa = kappa

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

    def dyna_q_plus(self, number_of_episodes, alpha=0.1, gamma=1., epsilon=0.1):
        steps_list = []
        for epi_iter in range(number_of_episodes):
            state = self.env.reset()
            step_num = 0
            while True:
                action = self.select_action(state)
                new_state, reward, is_done, _ = self.env.step(action)
                # q learning
                q_state_next = []
                for action_iter in range(self.env.action_space.n):
                    q_state_next.append(self.value_state_action[(new_state, action_iter)])
                q_state_next = max(q_state_next)
                q_state_current = self.value_state_action[(state, action)]
                self.value_state_action[(state, action)] = \
                    q_state_current + alpha * (reward + gamma * q_state_next - q_state_current)
                # update policy
                self.update_policy(state, epsilon)
                # model learning
                self.model[state][action] = [reward, new_state, self.total_step_num]
                if [state, action] not in self.model_state_action_list:
                    self.model_state_action_list.append([state, action])
                # planning
                if len(self.model_state_action_list) > 0:
                    for n_iter in range(self.n):
                        state_selected, action_selected = random.choice(self.model_state_action_list)
                        reward_in_model, new_state_in_model, last_update_time = \
                            self.model[state_selected][action_selected]
                        reward_in_model = reward_in_model + self.kappa * np.sqrt(self.total_step_num - last_update_time)
                        q_state_next_in_model = []
                        for action_iter in range(self.env.action_space.n):
                            q_state_next_in_model.append(self.value_state_action[(new_state_in_model, action_iter)])
                        q_state_next_in_model = max(q_state_next_in_model)
                        q_state_current_in_model = self.value_state_action[(state_selected, action_selected)]
                        self.value_state_action[(state_selected, action_selected)] = \
                            q_state_current_in_model + alpha * (reward_in_model + gamma * q_state_next_in_model -
                                                                q_state_current_in_model)
                        # update policy in planning
                        self.update_policy(state_selected, epsilon)
                if is_done:
                    break
                self.total_step_num += 1
                step_num += 1
                state = new_state
            steps_list.append(step_num)
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
    agent = Agent(env, n=10, kappa=0)
    dq_step_rewards_list = []
    steps = 0
    for i in range(400):
        agent.dyna_q_plus(1, alpha=0.1, gamma=0.95, epsilon=.3)
        dq_step_rewards_list.append([agent.total_step_num, i])
        # if i == 50:
        #     agent.env = GridWorld(6, [25, 26, 27, 28, 29], start_position=31, end_position_list=[5])
    dq_step_rewards_list = np.array(dq_step_rewards_list)
    plt.plot(dq_step_rewards_list[:, 0], dq_step_rewards_list[:, 1], label='Dyna_Q')
    plt.legend()
    plt.show()
