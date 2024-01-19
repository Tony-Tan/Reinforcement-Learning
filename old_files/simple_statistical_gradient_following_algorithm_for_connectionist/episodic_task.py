# Williams, Ronald J. “Simple Statistical Gradient-Following Algorithms for
# Connectionist Reinforcement Learning.” Machine Learning 8, no. 3 (1992): 229–56.
# episodic task
# learning_algorithms 13.1 short corridor with switched actions in reinforcement leanring: an introduction
# by Richard S. Sutton
import copy

import short_corridor_with_switched_actions as scwsa
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def squashing_function(s):
    return 1. / (1. + np.exp(-s))


class Agent:
    def __init__(self, seed):
        self.weight = -2.8
        self.np_random = np.random
        self.np_random.seed(seed)

    def select_action(self, x):
        # action_distribution = squashing_function(self.weight * x)
        action_distribution = squashing_function(self.weight)

        # action_distribution += EPSILON
        action = self.np_random.choice([0, 1], p=[1 - action_distribution, action_distribution])
        return action

    def run(self, repeat_time):
        env = scwsa.ShortCorridorWithSwitchedActions()
        alpha = 1e-4
        baseline = 0
        reward_log = []
        for epoch_i in range(repeat_time):
            trace = []
            state = env.reset()
            is_done = False
            action = self.select_action(state)
            reward_sum = 0
            while not is_done:
                next_state, reward, is_done, _ = env.step(action)
                trace.append([state, action, reward])
                reward_sum += reward
                state = next_state
                action = self.select_action(state)
            reward_log.append(copy.deepcopy(reward_sum))
            for t_i in trace:
                reward_sum = reward_sum - t_i[2]
                # x = t_i[0]
                x = 1
                s = self.weight * 1.
                y = t_i[1]
                p = 1. / (1. + np.exp(-s))
                delta_weight = (y-p)/(p*(1-p)) * np.exp(-s) / ((1 + np.exp(-s)) ** 2) * 1.
                self.weight += alpha*(reward_sum - baseline)*delta_weight
            # if epoch_i % 1000 == 0:
            #     print('epoch: '+str(epoch_i)+' left probability: ' + str(squashing_function(self.weight)))
        return np.array(reward_log)


def run_one(seed, episodic_num_per_experiment=10000):
    agent = Agent(seed)
    reward_log = agent.run(episodic_num_per_experiment)
    return reward_log


def experiments(thread_num=8):
    experiment_time = 1000
    episodic_num_per_experiment = 15000
    reward_matrix = []
    for experiment_i in range(int(experiment_time / thread_num)):
        print('experiment: '+str(experiment_i))
        result = []
        pool = Pool()
        seed_seq = np.random.randint(0, 100000, thread_num)
        for thread_i in range(thread_num):
            # reward_log = agent.run(1000)
            result.append(pool.apply_async(run_one, [seed_seq[thread_i], episodic_num_per_experiment]))
        pool.close()
        pool.join()
        for result_i in result:
            reward_thread = result_i.get(timeout=500)
            reward_matrix.append(reward_thread)
    reward_matrix = np.array(reward_matrix)
    average_reward = np.sum(reward_matrix, axis=0)/experiment_time
    plt.plot(average_reward)
    plt.show()


if __name__ == '__main__':
    experiments()





