import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import sys
import time


# log class
class Logger:
    def __init__(self, log_name: str, log_path: str = './', print_in_terminal: bool = True):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # remove illegal characters in log_name that cannot in a path and add time stamp
        log_name_ = log_name.replace('/', '-') + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log_file = open(os.path.join(log_path, log_name_ + '.log'), 'w+')
        self.tb_writer = SummaryWriter(log_dir=os.path.join(log_path, log_name_))
        self.print_in_terminal = print_in_terminal

    def tb_scalar(self, *args):
        """
        :param args:
        :return:
        """
        self.tb_writer.add_scalar(*args)

    def __del__(self):
        self.log_file.close()

    def msg(self, info: str):
        """
        :param info:
        :return:
        """
        time_strip = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        complete_info = '{time_strip:<10}: {info:<10}'.format(time_strip=time_strip, info=info)
        self.log_file.write(complete_info + '\n')
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

def debugger_time_cost(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} time cost: {end_time - start_time}')
        return result
    return wrapper

if __name__ == '__main__':
    logger_ = Logger(log_name='test_log.txt', log_path='./')
    # logger_('test logger...')
