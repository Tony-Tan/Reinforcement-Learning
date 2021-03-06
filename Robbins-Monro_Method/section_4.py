import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self):
        pass

    def generate(self, x):
        """
        set F(x) the uniform distribution over [0,1]
        :param x: an approximation generated by the method
        :return: a sample from the distribution
        """
        z = np.random.rand(1)[0]
        if z <= x:
            return 1
        else:
            return 0


class RobbinsMonroMethod:
    def __init__(self, repeat_times_=1000, method_type_='single', batch_size_=10):
        self.method_type = method_type_
        self.repeat_times = repeat_times_
        self.batch_size_ = batch_size_
        pass

    def run(self, alpha_):
        x = np.random.rand(1)[0]
        expectation_log = []
        if self.method_type == 'single':
            experiment = Experiment()
            for i in range(1, self.repeat_times):
                y = experiment.generate(x)
                x = x + 1 / i * (alpha - y)
                expectation_log.append(x)
        elif self.method_type == 'batch':
            experiment = Experiment()
            for i in range(1, self.repeat_times):
                y = [experiment.generate(x) for j in range(self.batch_size_)]
                y = np.array(y).mean()
                x = x + 1 / i * (alpha - y)
                expectation_log.append(x)
        return expectation_log


if __name__ == '__main__':
    alpha = 0.3
    rbm = RobbinsMonroMethod(method_type_='batch')
    expectation_log = rbm.run(alpha)
    # print('current expectation: ' + str(expectation_log[-1]))
    plt.plot(expectation_log, linewidth=1)
    plt.plot([1, 1000], [alpha, alpha], linestyle=":", linewidth=1)
    plt.show()