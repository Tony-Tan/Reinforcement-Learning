import matplotlib.pyplot as plt
import numpy as np
from environment.gride_world import GridWorld

import TebularPlanningAndLearning.Dyna_Q_plus as DQp

if __name__ == '__main__':
    episode_numbers = 400
    dqp_reward = 0.0
    average_times = 10
    average_step_array = np.zeros([2, episode_numbers])
    for ave_iter in range(average_times):
        env = GridWorld(6, [25, 26, 27, 28, 29], start_position=32, end_position_list=[5])
        agent_dqp = DQp.Agent(env, n=10, kappa=0.001)
        agent_dq = DQp.Agent(env, n=10, kappa=0)
        dqp_step_rewards_list = []
        dq_step_rewards_list = []
        for i in range(episode_numbers):
            agent_dqp.dyna_q_plus(1, alpha=0.1, gamma=0.95, epsilon=.3)
            dqp_step_rewards_list.append(agent_dqp.total_step_num)
            agent_dq.dyna_q_plus(1, alpha=0.1, gamma=0.95, epsilon=.3)
            dq_step_rewards_list.append(agent_dq.total_step_num)
            if i == episode_numbers / 2:
                agent_dqp.env = GridWorld(6, [25, 26, 27, 28], start_position=32, end_position_list=[5])
                agent_dq.env = GridWorld(6, [25, 26, 27, 28], start_position=32, end_position_list=[5])
        average_step_array[0] += np.array(dqp_step_rewards_list)
        average_step_array[1] += np.array(dq_step_rewards_list)
    plt.plot((average_step_array[0] - average_step_array[0][3]) / average_times,
             np.arange(1, episode_numbers + 1), label='Dyna_Q+')

    plt.plot((average_step_array[1] - average_step_array[1][3]) / average_times,
             np.arange(1, episode_numbers + 1), label='Dyna_Q')
    plt.legend()
    plt.show()
