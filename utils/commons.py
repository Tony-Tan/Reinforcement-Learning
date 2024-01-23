import numpy as np
import os
from datetime import datetime
import sys


# log class
class Logger:
    def __init__(self, log_name: str = '', log_path: str = './', print_in_terminal: bool = True):
        if log_name == '':
            log_name_ = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            log_name_ = log_name
        self.log_file = open(os.path.join(log_path, log_name_), 'a+')
        self.print_in_terminal = print_in_terminal
        pass

    def tensor_board(self, *args):
        pass

    def __del__(self):
        self.log_file.close()

    def __call__(self, info: str):
        """
        :param info:
        :return:
        """
        time_strip = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        complete_info = '{time_strip:<10}: {info:<10}'.format(time_strip=time_strip, info=info)
        self.log_file.write(complete_info+'\n')
        if self.print_in_terminal:
            print(complete_info)


# callbacks for training process
#
class Callbacks:
    def __init__(self):
        pass

    def on_episode_begin(self):
        pass

    def on_episode_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass


# exceptions
# class MethodNotImplement(Exception):
#     def __init__(self, info=None):
#         self.info = info
#
#     def __str__(self):
#         if self.info is None:
#             return 'The Method Has Not Been Implemented'
#         return self.info


class EnvNotExist(Exception):
    def __str__(self):
        return 'environment name or id not exist'


class PolicyNotImplement(Exception):
    def __str__(self):
        return 'Policy Has Not Been Implemented'


if __name__ == '__main__':
    logger_ = Logger(log_name='test_log.txt', log_path='./')
    logger_('test logger...')
