import collections
import matplotlib.pyplot as plt
import random
import math
import numpy as np


class Space:
    def __init__(self, n_list):
        self.n_list = n_list
        self.n = len(n_list)

    def __getitem__(self, item):
        return self.n_list[item]


class Sever:
    def __init__(self):
        self.is_free = True

    def is_available(self):
        return self.is_free

    def update_state(self):
        if not self.is_free:
            if random.randint(0,100) < 6:
                self.is_free = True

    def start_severing(self):
        self.is_free = False

    def reset(self):
        self.is_free = True


class QueuingTask:
    def __init__(self, customer_reward, customer_dis, sever_num=10):
        self.sever_num = sever_num
        self.customer_kind_number = len(customer_reward)
        self.customer_distributions = customer_dis
        self.customer_rewards_list = customer_reward
        self.sever_list = []
        for i in range(self.sever_num):
            s = Sever()
            self.sever_list.append(s)
        # reject or accept
        self.action_space = Space([0, 1])
        self.queue_head_priority = 0
        self.current_state = 0

    def reset(self):
        for sever in self.sever_list:
            sever.reset()
        self.queue_head_priority = np.random.choice(self.customer_kind_number, 1, p=self.customer_distributions)[0]
        self.current_state = (self.queue_head_priority, self.sever_num)
        return self.current_state

    def step(self, action):
        reward = 0
        # accept
        has_available_server = False
        for sever in self.sever_list:
            if sever.is_available():
                has_available_server = True
                break
        if action == 1 and has_available_server:
            for sever in self.sever_list:
                if sever.is_available():
                    sever.start_severing()
                    break
            reward = self.customer_rewards_list[self.queue_head_priority]
        # reject:
        # reward keep zero and update state

        # new customer at he head of the queue
        self.queue_head_priority = np.random.choice(self.customer_kind_number, 1, p=self.customer_distributions)[0]
        # update sever state
        available_sever_number = 0
        for sever in self.sever_list:
            sever.update_state()
            if sever.is_available():
                available_sever_number += 1
        self.current_state = (self.queue_head_priority, available_sever_number)
        return self.current_state, reward, False, {}


if __name__ == '__main__':
    rewards_list = [1, 2, 4, 8]
    rewards_distribution = [0.25, 0.25, 0.25, 0.25]
    env = QueuingTask(rewards_list, rewards_distribution)
    state = env.reset()
    print('initial state: ' + str(state))
    for i in range(100):
        action = random.randint(0, 1)
        state, r, is_done, _ = env.step(action)
        print('action: ' + str(action) + ' state: ' + str(state) + ' reward: ' + str(r))
