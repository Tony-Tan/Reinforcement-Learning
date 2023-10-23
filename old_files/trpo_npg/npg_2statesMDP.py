import matplotlib.pyplot as plt
import numpy as np
from environments import two_states_MDP as tsMDP
from multiprocessing import Pool


class NaturalPolicyGradient:
    def __init__(self, play_ground, trajectory_horizon):
        self.weights = np.array([-np.log(4), np.log(9)])
        # self.trajectory = []
        self.trajectory_horizon = trajectory_horizon
        self.env = play_ground
        # self.

    def sigmoid(self, x, a):
        bias = 5e-3
        return (1. - bias * 2) * (1.0 / (1.0 + np.exp(-(self.weights[0] * a))) * (1 - x) +
                                  1.0 / (1.0 + np.exp(-(self.weights[1] * a))) * x) + bias

    def next_action(self, x):
        pro_a0 = self.sigmoid(x, -1)
        if np.random.rand() >= pro_a0:
            return self.env.action_space[1]
        else:
            return self.env.action_space[0]

    def derivative(self, x, a):
        # delta_0 = 1/(1+exp(-a*w_0))
        # delta_1 = 1/(1+exp(-a*w_1))
        delta_0 = self.sigmoid(0, a)
        delta_1 = self.sigmoid(1, a)
        partial_w_0 = (1 - delta_0) * delta_0 * a * (1 - x)
        partial_w_1 = (1 - delta_1) * delta_1 * a * x
        return np.array([[partial_w_0], [partial_w_1]])

    def fisher_matrix(self, x, a):
        gradient = self.derivative(x, a)
        gradient_log = gradient / self.sigmoid(x, a)
        f_matrix = gradient_log.dot(gradient_log.transpose())
        return f_matrix

    def generate_trajectory(self):
        pass

    def run(self, alpha, random_seed, beta=0.9, natural_gradient=False):
        np.random.seed(random_seed)
        eta_array = []
        current_state = self.env.reset()
        eligibility_trace = np.zeros((2, 1))
        state_i_num = 0
        total_reward = 0
        fisher_matrix = np.eye(2)
        fisher_matrix_exp = np.zeros((2, 2))
        delta_eta = np.zeros((2, 1))
        horizon = 100
        print_horizon = 100000
        for i in range(1, 20000000):
            action = self.next_action(current_state)
            next_state, reward, is_done, _ = self.env.step(action)
            log_gradient = self.derivative(current_state, action) / self.sigmoid(current_state, action)
            if natural_gradient:
                fisher_matrix_exp = fisher_matrix_exp + log_gradient.dot(log_gradient.transpose())
            eligibility_trace = beta * eligibility_trace + log_gradient
            delta_eta += eligibility_trace * reward
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
                print('fisher matrix')
                print(fisher_matrix_exp / horizon)
                eta_array.append(total_reward / horizon)
            if i % horizon == 0:
                # current_state = self.env.reset()
                total_reward = 0
                eligibility_trace = np.zeros((2, 1))

                if natural_gradient:
                    delta_eta /= horizon
                    fisher_matrix_exp = fisher_matrix_exp / horizon + 1e-5 * np.eye(2)
                    fisher_matrix_inv = np.linalg.inv(fisher_matrix_exp)
                    fisher_matrix_inv /= np.max(fisher_matrix_inv)
                    weight_delta = alpha * fisher_matrix_inv.dot(delta_eta).transpose()[0]
                    self.weights += weight_delta
                    fisher_matrix_exp = np.zeros((2, 2))

                else:
                    delta_eta /= horizon
                    self.weights += alpha * delta_eta.transpose()[0]
                delta_eta = 0
        return eta_array


def policy_gradient(alpha, random_seed, natural_gradient_=True):
    play_ground = tsMDP.TwoStatesMDP([0.8, 0.2])
    # navilla policy gradient
    npg = NaturalPolicyGradient(play_ground, 50)
    eta_array_ = npg.run(.01, random_seed, natural_gradient=natural_gradient_)
    return eta_array_


def experiment():
    experiment_times = 8
    seed_seq = np.random.randint(0, 100000, experiment_times)
    thread_num = 8
    eta_matrix_natural = []
    eta_matrix_navilla = []

    for experiment_i in range(int(experiment_times / thread_num)):
        pool = Pool()
        for thread_i in range(thread_num):
            eta_matrix_navilla.append(pool.apply_async(policy_gradient,
                                                       [0.01, seed_seq[experiment_i * thread_num + thread_i], False]))

        pool.close()
        pool.join()

    for experiment_i in range(int(experiment_times / thread_num)):
        pool = Pool()
        for thread_i in range(thread_num):
            eta_matrix_natural.append(pool.apply_async(policy_gradient,
                                                       [0.01, seed_seq[experiment_i * thread_num + thread_i], True]))
        pool.close()
        pool.join()
    # prepare data
    eta_matrix_natural_result = []
    eta_matrix_navilla_result = []
    for eta_i in eta_matrix_navilla:
        eta_matrix_navilla_result.append(eta_i.get())
    for eta_i in eta_matrix_natural:
        eta_matrix_natural_result.append(eta_i.get())

    eta_matrix_navilla = np.array(eta_matrix_navilla_result)
    plt.plot(np.average(eta_matrix_navilla, axis=0), label='Navilla policy gradient, $\\alpha=0.01$')
    eta_matrix_natural = np.array(eta_matrix_natural_result)
    plt.plot(np.average(eta_matrix_natural, axis=0), label='Natural policy gradient, $\\alpha=0.01$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    experiment()
    # play_ground = tsMDP.TwoStatesMDP([0.8, 0.2])
    # # navilla policy gradient
    # eta_matrix_navilla = []
    # for epoch_i in range(5):
    #     print('========================%d=========================\n\n\n' % epoch_i)
    #     npg = NaturalPolicyGradient(play_ground, 50)
    #     eta_array_ = npg.run(.01, natural_gradient=False)
    #     eta_matrix_navilla.append(eta_array_)
    # eta_matrix_navilla = np.array(eta_matrix_navilla)
    # plt.plot(np.average(eta_matrix_navilla, axis=0), label='Navilla policy gradient, $\\alpha=0.01$')
    # # natural policy gradient
    # eta_matrix_natural = []
    # for epoch_i in range(5):
    #     print('========================%d=========================\n\n\n' % epoch_i)
    #     npg = NaturalPolicyGradient(play_ground, 50)
    #     eta_array_ = npg.run(.01, natural_gradient=True)
    #     eta_matrix_natural.append(eta_array_)
    # eta_matrix_natural = np.array(eta_matrix_natural)
    # plt.plot(np.average(eta_matrix_natural, axis=0), label='Natural policy gradient, $\\alpha=0.01$')
    #
    # plt.legend()
    # plt.show()
