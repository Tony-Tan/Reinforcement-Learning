import numpy as np
import random
import two_states_MDP as tsMDP


class NaturalPolicyGradient:
    def __init__(self, play_ground, trajectory_horizon):
        self.weights = np.array([-np.log(4), -np.log(9)])
        # self.trajectory = []
        self.trajectory_horizon = trajectory_horizon
        self.env = play_ground
        self.trans_matrix = np.zeros((2, 2))

    def sigmoid(self, x, a):
        return 1.0 / (1.0 + np.exp(-(self.weights[0] * a))) * (1 - x) + 1.0 / (1.0 + np.exp(-(self.weights[1] * a))) * x

    def next_action(self, x):
        pro_a0 = self.sigmoid(x, -1)
        if np.random.rand() > pro_a0:
            return self.env.action_space[1]
        else:
            return self.env.action_space[0]

    def derivative(self, x, a):
        if x == 0:
            forward_res = 1.0 / (1.0 + np.exp(-(self.weights[0] * a)))
            partial_w_0 = (1 - forward_res) * forward_res * a
            partial_w_1 = 0
            return np.array([[partial_w_0], [partial_w_1]])
        elif x == 1:
            forward_res = 1.0 / (1.0 + np.exp(-(self.weights[1] * a)))
            partial_w_0 = 0
            partial_w_1 = (1 - forward_res) * forward_res * a
            return np.array([[partial_w_0], [partial_w_1]])

    def gradient_log_pi(self, x, a):
        deriv = self.derivative(x, a)
        return deriv / self.sigmoid(x, a)

    # def Fisher_matrix(self):
    #     f_matrix = np.zeros((2, 2))
    #     state_distri = self.stationary_distribution()
    #     # state_distri = self.stationary_distribution_sim()
    #     for s_i in range(len(self.state_space)):
    #
    #         f_matrix_s = np.zeros((2, 2))
    #         for a_i in range(len(self.action_space)):
    #             x = self.state_space[s_i]
    #             a = self.action_space[a_i]
    #             p_a = self.sigmoid(x, a)
    #             log_p_w = np.array([[(1 - p_a) * x * a, (1 - p_a) * a]])
    #             log_p_w_t = log_p_w.transpose()
    #             f_matrix_s += log_p_w_t.dot(log_p_w) * p_a
    #         f_matrix += state_distri[s_i] * f_matrix_s
    #     return f_matrix + 1e-3 * np.eye(2)

    def generate_trajectory(self):
        current_state = self.env.reset()
        trajectory = []
        total_reward = 0
        for i in range(self.trajectory_horizon):
            action = self.next_action(current_state)
            next_state, reward, is_done, _ = self.env.step(action)
            trajectory.append([current_state, action, reward])
            total_reward += reward
            current_state = next_state
        return trajectory, total_reward

    def q_value(self, trajectory, total_reward):
        q_val = np.zeros((2, 2))
        exp_eta = total_reward / self.trajectory_horizon
        return_value = total_reward
        current_step = 0
        for step_i in trajectory:
            # state = 0,1
            # action = {-1,1}
            # convert action to {0,1}: int((action+2)/2)
            state, action, reward = step_i
            return_value -= reward

            if q_val[state][int((action + 2) / 2)] == 0:
                q_val[state][int((action + 2) / 2)] = return_value / (self.trajectory_horizon - current_step) - exp_eta
            if q_val.all() != 0:
                break
            current_step += 1
        return q_val

    def run(self, alpha):

        for i in range(1000000):
            delta_eta = 0
            trajectory, total_reward = self.generate_trajectory()
            q_val = self.q_value(trajectory, total_reward)
            for step_i in trajectory:
                state, action, reward = step_i
                policy_derivative = self.derivative(state, action)
                delta_eta += policy_derivative * q_val[state][int((action + 2) / 2)]
            # F = policy.Fisher_matrix()
            # delta_w = np.linalg.inv(F).dot(delta_eta)
            delta_w = delta_eta / len(trajectory)
            self.weights += alpha * delta_w.transpose()[0]
            if i % 10000 == 0:
                print('---------------%dth iteration--------------------' % i)
                print('weight:' + str(self.weights))
                state_i_num = 0
                for step_i in trajectory:
                    state, action, reward = step_i
                    if state == 0:
                        state_i_num += 1
                print('stationary distri' + str([state_i_num / self.trajectory_horizon,
                                                 1 - state_i_num / self.trajectory_horizon]))


if __name__ == '__main__':
    play_ground = tsMDP.TwoStatesMDP([0.8, 0.2])
    npg = NaturalPolicyGradient(play_ground, 20)
    npg.run(0.01)
