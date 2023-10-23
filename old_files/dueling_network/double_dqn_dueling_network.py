import gym
import random
from dqn.alg import AgentDQN
import copy
import dueling_network.Network as Network


class AgentDoubleDW(AgentDQN):
    def __init__(self, environment, mini_batch_size=32, episodes_num=100000,
                 k_frames=4, input_frame_size=84, memory_length=1e4, phi_temp_size=4,
                 model_path='./model/', log_path='./log/', learning_rate=0.00025, steps_c=10000):
        super(AgentDoubleDW, self).__init__(environment, mini_batch_size, episodes_num,
                                            k_frames, input_frame_size, memory_length, phi_temp_size,
                                            model_path, log_path, learning_rate, steps_c, algorithm_version='2015')
        self.state_action_value_function = Network.Net(4, self._action_n)
        self.target_state_action_value_function = Network.Net(4, self._action_n)
        self.state_action_value_function_temp = copy.deepcopy(self.state_action_value_function)
        self.state_action_value_function.to(self._device)
        self.target_state_action_value_function.to(self._device)
        self.load_existing_model()

    def learning(self, epsilon_max=1.0, epsilon_min=0.1, epsilon_decay=0.9999):
        """
        :param epsilon_max: float number, epsilon start number, 1.0 for most time
        :param epsilon_min: float number, epsilon end number, 0.1 in the paper
        :param epsilon_decay: float number, decay coefficient of epsilon
        :return: nothing
        """
        frame_num = self._phi_temp_size
        epsilon = epsilon_max
        for episode_i in range(1, self._episodes_num):
            # set a dynamic epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # switch tow sets parameters of network randomly
            if random.randint(0, 1):
                self.state_action_value_function_temp.load_state_dict(
                    self.state_action_value_function.state_dict())
                self.state_action_value_function.load_state_dict(
                    self.target_state_action_value_function.state_dict())
                self.target_state_action_value_function.load_state_dict(
                    self.state_action_value_function_temp.state_dict())

            frame_num_i, reward_i = self.learning_an_episode(epsilon)
            frame_num += frame_num_i
            self.record_reward(frame_num, reward_i, epsilon, episode_i)
            if episode_i % self._steps_c == 0:
                print('------------------------ updating target state action value function -----------------------')
                self.target_state_action_value_function.load_state_dict(self.state_action_value_function.state_dict())
            if episode_i % 500 == 0:
                self.save_model()


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    agent = AgentDoubleDW(env, steps_c=100)
    agent.learning(epsilon_max=1, epsilon_min=0.1)
