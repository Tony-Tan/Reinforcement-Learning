# Figure 2.3 shows the performance on the 10-armed bandit testbed of a greedy method using Q_1(a) = +5, for all a.

import matplotlib.pyplot as plt
import numpy as np
from environment.k_arm_bandit import KArmedBandit
from rich.progress import track

from epsilon_greedy import Agent


def experiment(total_step_num_=1000, repeat_experiment_n_times_=2000):
    optimal_action_percentage_q_0 = np.zeros(total_step_num_)
    optimal_action_percentage_q_5 = np.zeros(total_step_num_)
    for _ in track(range(repeat_experiment_n_times_), description="Repeating Experiment..."):
        env = KArmedBandit(np.random.normal(.0, 1.0, 10), np.ones(10))

        agent_q5 = Agent(env, 0, 5, 0.1)
        _, optimal_action_q5 = agent_q5.run(total_step_num_)
        optimal_action_percentage_q_5 += optimal_action_q5

        agent_q0 = Agent(env, 0.1, 0, 0.1)
        _, optimal_action_q0 = agent_q0.run(total_step_num_)
        optimal_action_percentage_q_0 += optimal_action_q0

    plt.figure(1)
    plt.plot(optimal_action_percentage_q_0 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='g',
             label='0.1-greedy initial_value=0')
    plt.plot(optimal_action_percentage_q_5 / repeat_experiment_n_times_, linewidth=1, alpha=0.7, c='b',
             label='greedy initial_value=5')
    plt.xlabel('time')
    plt.ylabel('% of optimal action')
    plt.legend()
    plt.savefig('./Figure/epsilon-greedy_initial_value_F2.3.png')
    plt.show()


if __name__ == '__main__':
    experiment(1000, 2000)
