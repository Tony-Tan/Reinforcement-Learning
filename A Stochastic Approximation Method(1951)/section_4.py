import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self):
        pass

    def generate(self, x):
        """
        set F(X) = X
        :param x:
        :return:
        """
        z = np.random.rand(1)[0]
        if z <= x:
            return 1
        else:
            return 0


if __name__ == '__main__':
    x = np.random.rand(1)[0]
    alpha = 0.3
    experiment = Experiment()
    expectation_log = []
    for i in range(1, 1000):
        y = experiment.generate(x)
        x = x + 1/i * (alpha - y)
        expectation_log.append(x)
    # print('current expectation: ' + str(expectation_log[-1]))
    plt.plot(expectation_log, linewidth=1)
    plt.plot([1, 1000], [alpha, alpha], linestyle=":", linewidth=1)
    plt.show()