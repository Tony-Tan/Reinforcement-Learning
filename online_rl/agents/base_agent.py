from collections import deque
import copy


class Agent:
    def __init__(self, env: EnvWrapper, replay_buffer_size: int, save_path: str, logger: Logger):
        self.env = env
        self.env4test = copy.deepcopy(env)
        self.logger = logger
        self.save_path = save_path
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def react(self, *args) -> np.ndarray:
        raise MethodNotImplement("Design the `act` method that returns the agent's action based on the current state.")

    def replay_buffer_append(self, transition: list):
        self.replay_buffer.append(transition)

    def learn(self, *args):
        raise MethodNotImplement("This method is responsible for training the agent. It takes the total number of "
                                 "time steps as input and updates the agent's policy and value function based on "
                                  "interactions with the environment.")

    def test(self, *args):
        raise MethodNotImplement("test model")

    def save(self):
        raise MethodNotImplement("store and restore agent parameters")

    def load(self):
        raise MethodNotImplement("store and restore agent parameters")


