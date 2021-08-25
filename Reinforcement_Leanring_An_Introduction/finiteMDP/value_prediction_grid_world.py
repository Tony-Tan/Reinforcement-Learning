# the solution in the book was calculated by solving linear equations
# the methods we using here is Dynamic Programming on page 75
import copy
import numpy as np
from environment.naive_grid_world import GridWorld


class Agent:
    def __init__(self, env_, gamma_=0.9):
        self.__env = env_
        self.__gamma = gamma_
        self.state_values_func = np.zeros(env_.state_space.n)
        self.__policy = np.ones(env_.action_space.n) / env_.action_space.n

    def predict(self):
        repeat_i = 0
        while True:
            last_step_state_value = copy.deepcopy(self.state_values_func)
            for state_i in self.__env.state_space:
                sum_ = 0.0
                for action_i in self.__env.action_space:
                    self.__env.set_current_state(state_i)
                    next_state, reward, _, _ = self.__env.step(action_i)
                    sum_ += self.__policy[action_i] * (reward + self.__gamma * self.state_values_func[next_state])
                self.state_values_func[state_i] = sum_
            # condition to stop the iteration
            # delta is small enough
            if np.min(np.abs(last_step_state_value - self.state_values_func)) < 0.00001 and repeat_i > 10:
                print('Prediction has finished! in %d loops'%repeat_i)
                return True
            repeat_i += 1


if __name__ == '__main__':
    env = GridWorld()
    agent = Agent(env)
    agent.predict()
    print(agent.state_values_func.reshape([5, 5]).round(1))
