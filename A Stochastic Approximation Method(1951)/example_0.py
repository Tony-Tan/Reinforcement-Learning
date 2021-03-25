import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    """
    the experiment is a generator who produce real number by
    $$
    y= \theta_2 \cdot x^2 +\theta_1 \cdot x + \theta_0
    $$
    where x is a r.v. having a uniform distribution on [0,1)
    """

    def __init__(self, generator_type):
        """
        :param generator_type:
            0: uniform distribution over  [0,1)
            1: Gaussian distribution over [0,1)
        """
        self.generator_type = generator_type
        pass

    def generator(self, theta):
        if self.generator_type == 0:
            x = np.random.rand(1)[0]
            return x * x + x + theta
        elif self.generator_type == 1:
            pass

    def expectation(self, theta):
        if self.generator_type == 0:
            return 1 * 1. / 3. + 1 / 2. + theta


if __name__ == '__main__':
    # initial theta
    alpha = 0.35
    theta = np.random.rand(1)[0]
    a = 1
    exp = Experiment(0)
    expectation_log = []
    for i in range(1, 2000):
        y = exp.generator(theta)
        theta = theta + 1.0 / i * (alpha - y)
        expectation_log.append(exp.expectation(theta))
        # print('current expectation: ' + str(expectation_log[-1]))
    plt.plot(expectation_log, linewidth=1)
    plt.plot([1, 2000], [alpha, alpha], linestyle=":", linewidth=1)
    plt.show()
