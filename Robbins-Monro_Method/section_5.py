import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self):
        self.simulate_beta_0_sum = 0.0
        self.simulate_beta_1_sum = 0.0
        self.simulate_beta_0_bias = np.random.rand(1)[0]
        self.simulate_beta_1_bias = np.random.rand(1)[0]
        self.simulate_times = 0
        pass

    def generate(self, x):
        """
        set F(x) the uniform distribution over [0,1]
        :param x: an approximation generated by the method
        :return: a sample from the distribution
        """
        beta_0 = np.random.rand(1)[0] + self.simulate_beta_0_bias + 1e-10
        beta_1 = np.random.rand(1)[0] + self.simulate_beta_1_bias + 1e-10
        self.simulate_beta_0_sum += beta_0
        self.simulate_beta_1_sum += beta_1
        self.simulate_times += 1
        return beta_0 + beta_1 * x

    def ground_truth(self, alpha_):
        if self.simulate_times < 100:
            print('you need run generate function more than 100 times to compute ground truth!')
            return 0
        return (alpha_ * self.simulate_times - self.simulate_beta_0_sum) / self.simulate_beta_1_sum


class RobbinsMonroMethod:
    def __init__(self, repeat_times_=1000, method_type_='single', batch_size_=10):
        self.method_type = method_type_
        self.repeat_times = repeat_times_
        self.batch_size_ = batch_size_
        pass

    def run(self, alpha_):
        x = np.random.rand(1)[0]
        expectation_log = []
        experiment = Experiment()

        if self.method_type == 'single':
            for i in range(1, self.repeat_times):
                y = experiment.generate(x)
                x = x + 1 / i * (alpha - y)
                expectation_log.append(x)
        elif self.method_type == 'batch':
            for i in range(1, self.repeat_times):
                y = [experiment.generate(x) for j in range(self.batch_size_)]
                y = np.array(y).mean()
                x = x + 1 / i * (alpha - y)
                expectation_log.append(x)
        ground_truth = experiment.ground_truth(alpha_)
        return expectation_log, ground_truth


if __name__ == '__main__':
    alpha = 0.4
    rbm = RobbinsMonroMethod(method_type_='single')
    expectation_log,gt = rbm.run(alpha)
    # print('current expectation: ' + str(expectation_log[-1]))
    plt.plot(expectation_log, linewidth=1)
    plt.plot([1, 1000], [gt, gt], linestyle=":", linewidth=1)
    plt.show()