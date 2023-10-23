from exps import dqn as dqn
import os
import argparse
from rl_algorithms.agents.base_agent import Agent
from rl_algorithms.environments.env import Env
from rl_algorithms.common.core import *

parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='ALE/Pong-v5', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--mini_batch_size', default=32, type=int, help='ccn training batch size，default: 32')
parser.add_argument('--episodes_num', default=100000, type=int, help='max training episodes，default: 100000')
parser.add_argument('--k_frames', default=4, type=int, help='dqn skip k frames each step，default: 4')
parser.add_argument('--input_frame_width', default=84, type=int, help='cnn input image width, default: 84')
parser.add_argument('--input_frame_height', default=110, type=int, help='cnn input image height，default: 110')
parser.add_argument('--memory_length', default=15000, type=int, help='memory buffer size ，default: 15000')
parser.add_argument('--init_data_size', default=2000, type=int, help='min data size before training cnn ，default: 2000')
parser.add_argument('--gamma', default=0.99, type=float, help='value decay, default: 0.99')
parser.add_argument('--model_path', default='./model/', type=str, help='model save path ，default: ./model/')
parser.add_argument('--log_path', default='./log/', type=str, help='log save path，default: ./log/')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='cnn learning rate，default: 1e-5')
parser.add_argument('--steps_c', default=100, type=int,
                    help='synchronise target value network periods，default: 100')
parser.add_argument('--epsilon_min', default=0.1, type=float,
                    help='min epsilon of epsilon-greedy，default: 0.1')
parser.add_argument('--epsilon_for_test', default=0.05, type=float,
                    help='epsilon of epsilon-greedy for testing agent，default: 0.05')
parser.add_argument('--model_saving_period', default=80000, type=int,
                    help='model saving period(step)，default: 80000')
args = parser.parse_args()

def test():
    pass

def training(dqn_agent: Agent, max_episodes:int,max_steps:int ):
    obs = dqn_agent.env.reset()
    epsilon = 1
    learning_episodes = 0
    learning_steps = 0
    while learning_episodes < max_episodes and learning_steps < max_steps:
        action = dqn_agent.react(obs, epsilon)
        obs_next, reward, terminated, truncated, info = dqn_agent.env.step(action)
        dqn_agent.replay_buffer_append([obs_next, reward, terminated, truncated, info])
        dqn_agent.learn()
        learning_steps += 1
        if terminated or truncated:
            obs = dqn.env.reset()
            learning_episodes += 1
        else:
            obs = obs_next
        if learning_steps % args.test_period_steps == 0:
            test()


if __name__ == '__main__':

    env_name = "ALE/Boxing-v5"
    env_ = Env(env_name)
    env_ = dqn.DQNGym(env_name)
    agent = dqn.AgentDQN(env_.action_dim, os.path.join('model/'))
    agent.load('./model/03-16-11-07-57_ALE_Boxing-v5/03-17-08-11-22_value.pth')
    agent.epsilon = 0.00001
    dqn_play_ground = dqn.DQNPlayGround(env_, agent)
    frame_num = 0
    last_record_episode = None
    frame_num_last_record = 0
    while True:
        dqn_play_ground.play_rounds(1000, display=True)
        frame_num += 1