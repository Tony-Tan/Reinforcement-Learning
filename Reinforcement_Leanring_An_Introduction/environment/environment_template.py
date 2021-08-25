# this module is the template of all environment in this project
# some basic and common features are listed in this class
class ENVError(Exception):
    def __init__(self, value_):
        self.__value = value_

    def __str__(self):
        print('Environment error occur: ' + self.__value)


class ENV:
    def __init__(self):
        pass

    def reset(self):
        raise ENVError('You have not defined the reset function of environment')

    def step(self, action):
        raise ENVError('You have not defined the step function of environment')
