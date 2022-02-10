import matplotlib.pyplot as plt
import numpy as np
import random
import two_states_MDP as tsMDP


class NaturalPolicyGradient:
    def __init__(self, play_ground, trajectory_horizon):
        self.weights = np.array([-np.log(4), np.log(9)])
        # self.trajectory = []
        self.trajectory_horizon = trajectory_horizon
        self.env = play_ground
        # self.

    def sigmoid(self, x, a):
        return 1.0 / (1.0 + np.exp(-(self.weights[0] * a))) * (1 - x) + 1.0 / (1.0 + np.exp(-(self.weights[1] * a))) * x

    def next_action(self, x):
        pro_a0 = self.sigmoid(x, -1)
        if np.random.rand() > pro_a0:
            return self.env.action_space[1]
        else:
            return self.env.action_space[0]

    def derivative(self, x, a):
        forward_res_0 = 1.0 / (1.0 + np.exp(-(self.weights[0] * a)))
        forward_res_1 = 1.0 / (1.0 + np.exp(-(self.weights[1] * a)))
        partial_w_0 = (1 - forward_res_0) * forward_res_0 * a * (1-x)
        partial_w_1 = (1 - forward_res_1) * forward_res_1 * a * x
        return np.array([[partial_w_0], [partial_w_1]])

    def gradient_log_pi(self, x, a):
        deriv = self.derivative(x, a)
        return deriv / self.sigmoid(x, a)

    def Fisher_matrix(self,x,a):
        gradient_log = self.derivative(x,a)/self.sigmoid(x,a)
        f_matrix = gradient_log.dot(gradient_log.transpose())
        return f_matrix

    def generate_trajectory(self):
        pass
        #, total_reward

    # def q_value(self, trajectory, total_reward):
    #     q_val = np.zeros((2, 2))
    #     exp_eta = total_reward / self.trajectory_horizon
    #     return_value = total_reward
    #     current_step = 0
    #     for step_i in trajectory:
    #         # state = 0,1
    #         # action = {-1,1}
    #         # convert action to {0,1}: int((action+2)/2)
    #         state, action, reward = step_i
    #         if q_val[state][int((action + 2) / 2)] == 0:
    #             q_val[state][int((action + 2) / 2)] = return_value / (self.trajectory_horizon - current_step) - exp_eta
    #         if q_val.all() != 0:
    #             break
    #         return_value -= reward
    #         current_step += 1
    #     return q_val

    def run(self, alpha, beta=0.9,natural_gradient=True):
        eta_array = []
        current_state = self.env.reset()
        eligibility_trace = np.zeros((2, 1))
        state_i_num = 0
        total_reward = 0
        fisher_matrix = np.eye(2)
        fisher_matrix_exp = np.zeros((2,2))
        delta_eta = 0
        horizon = 10000
        print_horizon = 100000
        for i in range(1, 10000000):
            # if i % self.trajectory_horizon == 0:
            # F = policy.Fisher_matrix()
            # delta_w = np.linalg.inv(F).dot(delta_eta)
            # delta_w = delta_eta / self.trajectory_horizon
            # self.weights += alpha * delta_w.transpose()[0]
            # delta_eta = 0
            # eligibility_trace = np.zeros((2, 1))

            action = self.next_action(current_state)
            next_state, reward, is_done, _ = self.env.step(action)
            if natural_gradient:
                fisher_matrix_exp += self.Fisher_matrix(current_state, action)
                eligibility_trace = self.derivative(current_state, action) / self.sigmoid(current_state, action)
                delta_eta += eligibility_trace * reward

            else:
                eligibility_trace = beta * eligibility_trace + \
                                    self.derivative(current_state, action) / self.sigmoid(current_state, action)
                delta_eta = eligibility_trace * reward
                self.weights += alpha * delta_eta.transpose()[0]

            total_reward += reward
            if current_state == 0:
                state_i_num += 1
            current_state = next_state
            if i % print_horizon == 0:
                trans_matrix = np.zeros((2, 2))
                trans_matrix[0][0] = self.sigmoid(0, -1)
                trans_matrix[0][1] = self.sigmoid(0, 1)
                trans_matrix[1][0] = self.sigmoid(1, 1)
                trans_matrix[1][1] = self.sigmoid(1, -1)
                print('---------------%dth iteration--------------------' % i)
                print(trans_matrix)
                print('weight:' + str(self.weights))
                print('stationary distri' + str([state_i_num / i, 1 - state_i_num / i]))
                print('eta:' + str(total_reward / horizon))
            if i % horizon == 0:
                current_state = self.env.reset()
                eligibility_trace = np.zeros((2, 1))
                eta_array.append(total_reward / horizon)
                total_reward = 0

                if natural_gradient:
                    delta_eta /= horizon
                    fisher_matrix_exp = fisher_matrix_exp/horizon + 1e-3 * np.eye(2)
                    self.weights += alpha * np.linalg.inv(fisher_matrix_exp).dot(delta_eta).transpose()[0]
                    fisher_matrix_exp = np.zeros((2, 2))
                    delta_eta = 0
        return eta_array


if __name__ == '__main__':
    play_ground = tsMDP.TwoStatesMDP([0.8, 0.2])
    eta_matrix = []
    for epoch_i in range(1):
        print('========================%d=========================\n\n\n'%epoch_i)
        npg = NaturalPolicyGradient(play_ground, 50)
        eta_array_ = npg.run(.1)
        eta_matrix.append(eta_array_)
    eta_matrix = np.array(eta_matrix)
    plt.plot(np.average(eta_matrix, axis=0))
    plt.show()
