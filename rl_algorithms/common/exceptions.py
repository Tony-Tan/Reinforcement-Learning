# exceptions
class MethodNotImplement(Exception):
    def __init__(self, info=None):
        self.info = info

    def __str__(self):
        if self.info is None:
            return 'The Method Has Not Been Implemented'
        return self.info


class EnvNotExist(Exception):
    def __str__(self):
        return 'environment name or id not exist'
