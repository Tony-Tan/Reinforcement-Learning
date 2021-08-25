# exercise 2.5 on page 33 of the 2nd edition Design and conduct an experiment to demonstrate the difficulties that
# sample-average methods have for non-stationary problems. Use a modified version of the 10-armed testbed in which
# all the q_{\star}(a) start out equal and then take independent random walks (say by adding a normally distributed
# increment with mean zero and standard deviation 0.01 to all the q_{\star}(a) on each step). Prepare plots like
# Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value
# method using a constant step-size parameter, \alpha= 0.1. Use \epsilon= 0.1 and longer runs, say of 10,000 steps.


import matplotlib.pyplot as plt
import numpy as np
from environment.k_arm_bandit import KArmedBanditRW
from rich.progress import track

from epsilon_greedy import Agent


def experiment(total_step_num_, repeat_experiment_n_times_):
    average_reward_0 = np.zeros(total_step_num_)
    optimal_action_percentage_0 = np.zeros(total_step_num_)
    for _ in track(range(repeat_experiment_n_times_), description="Repeating Experiment..."):
        env = KArmedBanditRW(np.random.normal(.0, 1.0, 10), np.ones(10))
        agent_0 = Agent(env, 0.1, 0, 0.1)
        reward_0, optimal_action_0 = agent_0.run(total_step_num_)
        average_reward_0 += reward_0
        optimal_action_percentage_0 += optimal_action_0
    plt.figure(1)
    plt.plot(average_reward_0 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='0.1-greedy $\\alpha=0.1$ initial_value=0')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.figure(2)
    plt.plot(optimal_action_percentage_0 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='0.1-greedy $\\alpha=0.1$ initial_value=0')
    plt.xlabel('Steps')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    experiment(10000, 2000)
