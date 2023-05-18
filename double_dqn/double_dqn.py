import gym
import random
from dqn.dqn import *
import copy


class DoubleDQN(DQN):
    def __init__(self, action_dim: int, save_path: str):
        super(DoubleDQN, self).__init__(action_dim, save_path)
        self.value_nn_temp = copy.deepcopy(self.value_nn)

    def learn(self):
        if random.randint(0, 1):
            self.value_nn_temp.load_state_dict(
                self.value_nn.state_dict())
            self.value_nn.load_state_dict(
                self.target_value_nn.state_dict())
            self.target_value_nn.load_state_dict(
                self.value_nn_temp.state_dict())
        super().learn()

def execute():
    now = int(round(time.time() * 1000))
    now02 = time.strftime('%m-%d-%H-%M-%S', time.localtime(now / 1000))
    env_name = "ALE/Pong-v5"
    env_ = DQNGym(env_name)
    exp_name = now02 + "_" + env_name.replace('/', '_')
    agent = DQN(env_.action_dim, os.path.join('./model/', exp_name))
    training = Training(env_, agent, max_episodes=args.episodes_num)
    training.online_training()


if __name__ == '__main__':
    execute()