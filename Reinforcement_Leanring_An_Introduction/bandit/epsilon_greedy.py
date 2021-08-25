import time

import matplotlib.pyplot as plt
import numpy as np
from environment.k_arm_bandit import KArmedBandit
from rich.progress import track

from basic_moduls.epsilon_greedy import EpsilonGreedy


class Agent:
    def __init__(self, env_, epsilon_, initial_value_, step_size_):
        self._initial_value = initial_value_
        self._step_size = step_size_
        self._env = env_
        self._k = env_.action_space.n
        self._optimal_action = env_.optimal_action
        self._policy = EpsilonGreedy(epsilon_)

    def select_action(self, action_value):
        prob_distribution = self._policy(action_value)
        action = np.random.choice(self._k, 1, p=prob_distribution)
        return action[0]

    def run(self, total_step_num_):
        reward_array = np.zeros(total_step_num_)
        optimal_action_selected = np.zeros(total_step_num_)
        action_value_estimate = np.ones(self._k) * self._initial_value
        action_value_estimated_times = np.zeros(self._k)
        # environment reset. although it is useless here
        # keep it for a good habit
        state = self._env.reset()
        for step_i in range(total_step_num_):
            action = self.select_action(action_value_estimate)
            state, reward, is_done, _ = self._env.step(action)
            action_value_estimated_times[action] += 1
            # update
            if self._step_size == '1/n':
                step_size = 1. / action_value_estimated_times[action]
            else:
                step_size = self._step_size
            # pseudocode on page 32
            error_in_estimation = (reward - action_value_estimate[action])
            action_value_estimate[action] = action_value_estimate[action] + step_size * error_in_estimation

            reward_array[step_i] = reward
            if action in self._optimal_action:
                optimal_action_selected[step_i] = 1
        return reward_array, optimal_action_selected


def experiment(total_step_num_=1000, repeat_experiment_n_times_=2000):
    average_reward_0 = np.zeros(total_step_num_)
    average_reward_0_1 = np.zeros(total_step_num_)
    average_reward_0_01 = np.zeros(total_step_num_)
    optimal_action_percentage_0 = np.zeros(total_step_num_)
    optimal_action_percentage_0_1 = np.zeros(total_step_num_)
    optimal_action_percentage_0_01 = np.zeros(total_step_num_)
    start_time = time.time()
    for _ in track(range(repeat_experiment_n_times_), description="Repeating Experiment..."):
        env = KArmedBandit(np.random.normal(.0, 1.0, 10), np.ones(10))
        agent_0 = Agent(env, 0, 0, '1/n')
        reward_0, optimal_action_0 = agent_0.run(total_step_num_)
        average_reward_0 += reward_0
        optimal_action_percentage_0 += optimal_action_0

        agent_0_1 = Agent(env, 0.1, 0, '1/n')
        reward_0_1, optimal_action_0_1 = agent_0_1.run(total_step_num_)
        average_reward_0_1 += reward_0_1
        optimal_action_percentage_0_1 += optimal_action_0_1

        agent_0_01 = Agent(env, 0.01, 0, '1/n')
        reward_0_01, optimal_action_0_01 = agent_0_01.run(total_step_num_)
        average_reward_0_01 += reward_0_01
        optimal_action_percentage_0_01 += optimal_action_0_01
    end_time = time.time()
    print('total time: ' + str(end_time - start_time))
    # draw results
    plt.figure(1)
    plt.plot(average_reward_0 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='0-greedy initial_value=0')
    plt.plot(average_reward_0_1 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='b',
             label='0.1-greedy initial_value=0')
    plt.plot(average_reward_0_01 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='r',
             label='0.01-greedy initial_value=0')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.savefig('./Figure/epsilon-greedy_F2.2.0.png')
    plt.figure(2)
    plt.plot(optimal_action_percentage_0 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='0-greedy initial_value=0')
    plt.plot(optimal_action_percentage_0_1 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='b',
             label='0.1-greedy initial_value=0')
    plt.plot(optimal_action_percentage_0_01 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='r',
             label='0.01-greedy initial_value=0')
    plt.xlabel('steps')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.savefig('./Figure/epsilon-greedy_F2.2.1.png')
    plt.show()


# for figure 2.2
if __name__ == '__main__':
    experiment()
