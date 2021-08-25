class BasicConceptError(Exception):
    def __init__(self, value_):
        self.__value = value_

    def __str__(self):
        print('[Basic Concepts Error!]: ' + self.__value)


class Space:
    def __init__(self, initial_list):
        if not isinstance(initial_list, list):
            raise BasicConceptError('Space should be initialized by a list!')
        if len(initial_list) == 0:
            raise BasicConceptError('Empty space may destroy your whole project!')
        self.__list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.__list[index]


if __name__ == '__main__':
    action_space = Space(1)
