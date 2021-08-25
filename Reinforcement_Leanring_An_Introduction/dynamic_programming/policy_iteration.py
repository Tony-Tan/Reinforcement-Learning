# Example 4.2: Jack’s Car Rental

from iterative_policy_evaluation import Agent as IPE_Agent
import numpy as np
import copy

class Agent(IPE_Agent):
    def __init__(self, env_, gamma_=1.0):
        super().__init__(env_, gamma_)

    def run(self):
        while True:
            self.predict()
            policy_stable = True
            for state_i in range(self.state_values_func.size()):
                old_action = copy.deepcopy(self.policy[state_i])
                state_action_value = np.zeros(self.env.action_space.n)
                for action_i in self.env.action_space:
                    self.env.set_current_state(state_i)
                    next_state, reward, _, _ = self.env.step(action_i)
                    state_action_value[action_i] = reward + self.gamma * self.state_values_func[next_state]
                optimal_action = np.random.choice(
                    np.flatnonzero(state_action_value == state_action_value.max()))
                np.zeors(self.policy[state_i])
                self.policy[state_i][optimal_action] = 1.0
                if old_action != self.policy:
                    policy_stable = False
            if policy_stable:
                return
