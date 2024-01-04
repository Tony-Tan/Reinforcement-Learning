import numpy as np
import os
from datetime import datetime
import sys


# log class
class Logger:
    def __init__(self, log_path: str, print_in_terminal: bool = True):
        self.log_file = open(os.path.join(log_path), 'a+')
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


if __name__ == '__main__':
    logger_ = Logger('./test_log.txt')
    logger_('test logger...')
