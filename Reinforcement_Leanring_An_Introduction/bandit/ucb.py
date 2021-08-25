import matplotlib.pyplot as plt
import numpy as np
from environment.k_arm_bandit import KArmedBandit
from rich.progress import track

from bandit.epsilon_greedy import Agent as EG_Agent
from basic_moduls.ucb import UCB


class Agent():
    def __init__(self, env_, c_, initial_value_, step_size_):
        self._initial_value = initial_value_
        self._step_size = step_size_
        self._env = env_
        self._k = env_.action_space.n
        self._optimal_action = env_.optimal_action
        self._policy = UCB(c_)

    def select_action(self, action_value_, current_step_num_, action_selected_num_array_):
        prob_distribution = self._policy(action_value_, current_step_num_, action_selected_num_array_)
        action = np.random.choice(self._k, 1, p=prob_distribution)
        return action[0]

    def run(self, total_step_num_):
        reward_array = np.zeros(total_step_num_)
        optimal_action_array = np.zeros(total_step_num_)
        action_value_estimate = np.ones(self._k) * self._initial_value
        action_value_estimated_times = np.zeros(self._k)
        # environment reset. although it is useless here
        # keep it for a good habit
        state = self._env.reset()
        for step_i in range(total_step_num_):
            action = self.select_action(action_value_estimate, step_i + 1, action_value_estimated_times)
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
                optimal_action_array[step_i] = 1
        return reward_array, optimal_action_array


def experiment(total_step_num_, repeat_experiment_n_times_):
    average_reward_ubc = np.zeros(total_step_num_)
    optimal_action_percentage_ubc = np.zeros(total_step_num_)
    average_reward_eg = np.zeros(total_step_num_)
    optimal_action_percentage_eg = np.zeros(total_step_num_)
    for _ in track(range(repeat_experiment_n_times_), description="Repeating Experiment..."):
        env = KArmedBandit(np.random.normal(.0, 1.0, 10), np.ones(10))
        agent_ubc = Agent(env, 2, 0, '1/n')
        reward_ubc, optimal_action_ubc = agent_ubc.run(1000)
        average_reward_ubc += reward_ubc
        optimal_action_percentage_ubc += optimal_action_ubc

        agent_epsilon_0_1 = EG_Agent(env, 0.1, 0, '1/n')
        reward_eg, optimal_action_eg = agent_epsilon_0_1.run(1000)
        average_reward_eg += reward_eg
        optimal_action_percentage_eg += optimal_action_eg

    plt.figure(1, figsize=(12, 10))
    plt.plot(average_reward_ubc / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='UCB initial_value=0')
    plt.plot(average_reward_eg / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='b',
             label='0.1-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.legend()
    plt.savefig('./Figure/UCB_reward_F2.4.png')
    plt.figure(2, figsize=(12, 10))
    plt.plot(optimal_action_percentage_ubc / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='UCB initial_value=0')
    plt.plot(optimal_action_percentage_eg / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='b',
             label='0.1-greedy initial_value=0')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.savefig('./Figure/UCB_optimal_F2.4.png')
    plt.show()


# for figure 2.2
if __name__ == '__main__':
    experiment(1000, 2000)
