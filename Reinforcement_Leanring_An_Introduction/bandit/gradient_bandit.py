# Figure 2.5 shows results with the gradient bandit algorithm on a variant of the 10armed testbed in which the true
# expected rewards were selected according to a normal distribution with a mean of +4 instead of zero (and with unit
# variance as before).

import matplotlib.pyplot as plt
import numpy as np
from environment.k_arm_bandit import KArmedBandit
from rich.progress import track


class Agent:
    def __init__(self, env_, step_size_):
        self._step_size = step_size_
        self._env = env_
        self._k = env_.action_space.n
        self._optimal_action = env_.optimal_action
        self._preference = np.zeros(self._k)
        self._policy = np.ones(self._k) / self._k

    def policy_update(self):
        total_preference = 0
        for i in self._env.action_space:
            total_preference += np.exp(self._preference[i])
        for i in self._env.action_space:
            self._policy[i] = np.exp(self._preference[i]) / total_preference

    def select_action(self):
        prob_distribution = self._policy
        action = np.random.choice(self._k, 1, p=prob_distribution)
        return action[0]

    def run(self, total_step_num_, baseline_=True):
        total_reward = 0
        reward_array = np.zeros(total_step_num_)
        optimal_action_selected = np.zeros(total_step_num_)
        # environment reset. although it is useless here
        # keep it for a good habit
        state = self._env.reset()
        for step_i in range(total_step_num_):
            action = self.select_action()
            state, reward, is_done, _ = self._env.step(action)
            total_reward += reward
            # pseudocode on page 32
            for action_i in self._env.action_space:
                average_reward = 0
                if baseline_:
                    average_reward = total_reward / (step_i + 1)
                if action_i == action:
                    self._preference[action_i] += self._step_size * (reward - average_reward) * \
                                                  (1. - self._policy[action_i])
                else:
                    self._preference[action_i] -= self._step_size * (reward - average_reward) * \
                                                  self._policy[action_i]
            self.policy_update()
            reward_array[step_i] = reward
            if action in self._optimal_action:
                optimal_action_selected[step_i] = 1
        return reward_array, optimal_action_selected


def experiment(total_step_num_=1000, repeat_experiment_n_times_=1000):
    optimal_action_percentage_4_0_1 = np.zeros(total_step_num_)
    optimal_action_percentage_4_0_4 = np.zeros(total_step_num_)
    optimal_action_percentage_0_0_1 = np.zeros(total_step_num_)
    optimal_action_percentage_0_0_4 = np.zeros(total_step_num_)
    for _ in track(range(repeat_experiment_n_times_), description="Repeating Experiment..."):
        env = KArmedBandit(np.random.normal(4, 1, 10), np.ones(10))
        agent_0_1 = Agent(env, 0.1)
        agent_0_4 = Agent(env, 0.4)
        _, optimal_action_4_0_1 = agent_0_1.run(total_step_num_, baseline_=True)
        optimal_action_percentage_4_0_1 += optimal_action_4_0_1
        _, optimal_action_4_0_4 = agent_0_4.run(total_step_num_, baseline_=True)
        optimal_action_percentage_4_0_4 += optimal_action_4_0_4

        agent_0_1 = Agent(env, 0.1)
        agent_0_4 = Agent(env, 0.4)
        _, optimal_action_0_0_1 = agent_0_1.run(total_step_num_, baseline_=False)
        optimal_action_percentage_0_0_1 += optimal_action_0_0_1
        _, optimal_action_0_0_4 = agent_0_4.run(total_step_num_, baseline_=False)
        optimal_action_percentage_0_0_4 += optimal_action_0_0_4

    # draw results
    plt.figure(1)
    plt.plot(optimal_action_percentage_4_0_1 / repeat_experiment_n_times_,
             linewidth=1, alpha=0.7, c='blue', label='Baseline=4 $\\alpha=0.1$')
    plt.plot(optimal_action_percentage_4_0_4 / repeat_experiment_n_times_,
             linewidth=1, alpha=0.7, c='lightblue', label='Baseline=4 $\\alpha=0.4$')
    plt.plot(optimal_action_percentage_0_0_1 / repeat_experiment_n_times_,
             linewidth=1, alpha=0.7, c='saddlebrown', label='Baseline=0 $\\alpha=0.1$')
    plt.plot(optimal_action_percentage_0_0_4 / repeat_experiment_n_times_,
             linewidth=1, alpha=0.7, c='rosybrown', label='Baseline=0 $\\alpha=0.4$')
    plt.xlabel('Steps')
    plt.ylabel('% of optimal action')
    plt.legend()
    # plt.savefig('./Figure/epsilon-greedy_F2.2.1.png')
    plt.show()


if __name__ == '__main__':
    experiment()
