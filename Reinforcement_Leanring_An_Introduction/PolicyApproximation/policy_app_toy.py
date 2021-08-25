import matplotlib.pyplot as plt
import numpy as np
from environment.corridor_gridworld import ShortCorridor


class Agent:
    def __init__(self, env):
        self.env = env

    def play(self, number_of_episodes, prob_to_right=0):
        reward_cumulate = 0
        for _ in range(number_of_episodes):
            self.env.reset()
            while True:
                action = np.random.choice(env.action_space.n, 1, p=[1. - prob_to_right, prob_to_right])
                action = action[0]
                new_state, reward, is_done, _ = self.env.step(action)
                reward_cumulate += reward
                if is_done:
                    break
        return float(reward_cumulate / number_of_episodes)


if __name__ == '__main__':
    env = ShortCorridor()
    agent = Agent(env)
    steps = []
    x_axis = []
    for i in range(150, 350):
        print('probability: ', i / 500.)
        x_axis.append(i / 500)
        steps.append(agent.play(500, i / 500.))
    plt.plot(x_axis, steps, alpha=0.7)
    plt.show()
