import argparse
from agents.dqn_agent import *
from environments.env_wrapper import EnvWrapper
from exploration.epsilon_greedy import *
import copy
import cv2
from tqdm import tqdm
from multiprocessing import Process, Queue, set_start_method


parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='Pong-v0', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--mini_batch_size', default=32, type=int,
                    help='cnn training batch size，default: 32')
parser.add_argument('--batch_num_per_epoch', default=500_000, type=int,
                    help='each epoch contains how many updates，default: 500,000')
parser.add_argument('--replay_buffer_size', default=1_000_000, type=int,
                    help='memory buffer size ，default: 200,000')
parser.add_argument('--training_episodes', default=100_000, type=int,
                    help='max training episodes，default: 100,000')
parser.add_argument('--skip_k_frame', default=4, type=int,
                    help='dqn skip k frames each step，default: 4')
parser.add_argument('--phi_channel', default=4, type=int,
                    help='phi temp size, default: 4')
parser.add_argument('--device', default='cuda', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--input_frame_width', default=84, type=int,
                    help='cnn input image width, default: 84')
parser.add_argument('--input_frame_height', default=84, type=int,
                    help='cnn input image height，default: 84')
parser.add_argument('--replay_start_size', default=100, type=int,
                    help='min data size before training cnn ，default: 6000')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='value decay, default: 0.99')
parser.add_argument('--no_op', default=30, type=int,
                    help='random action in the first no_op actions, default: 30')
parser.add_argument('--save_path', default='./data_log/', type=str,
                    help='model save path ，default: ./model/')
parser.add_argument('--log_path', default='../exps/dqn/', type=str,
                    help='log save path，default: ./log/')
parser.add_argument('--learning_rate', default=0.00001, type=float,
                    help='cnn learning rate，default: 0.00001')
parser.add_argument('--step_c', default=100_000, type=int,
                    help='synchronise target value network periods，default: 100')
parser.add_argument('--epsilon_max', default=1., type=float,
                    help='max epsilon of epsilon-greedy，default: 1.')
parser.add_argument('--epsilon_min', default=0.1, type=float,
                    help='min epsilon of epsilon-greedy，default: 0.1')
parser.add_argument('--exploration_steps', default=8_000_000, type=int,
                    help='min epsilon of epsilon-greedy，default: 1,000,000')
parser.add_argument('--epsilon_for_test', default=0.05, type=float,
                    help='epsilon of epsilon-greedy for testing agent，default: 0.05')
parser.add_argument('--agent_test_episodes', default=100, type=int,
                    help='agent test episode，default: 10')
parser.add_argument('--agent_saving_period', default=80000, type=int,
                    help='agent saving period(episode)，default: 80000')
args = parser.parse_args()


def test(agent: DQNAgent, test_episodes: int):
    env = EnvWrapper(args.env_name)
    exploration_method = EpsilonGreedy(args.epsilon_for_test)
    reward_cum = 0
    step_cum = 0
    for i in range(test_episodes):
        state, _ = env.reset()
        done = truncated = False
        step_i = 0
        while (not done) and (not truncated):
            obs = agent.perception_mapping(state, step_i)
            action = agent.select_action(obs, exploration_method)
            next_state, reward, done, truncated, inf = env.step(action)

            reward_cum += reward
            state = next_state
            step_i += 1
        step_cum += step_i
    return reward_cum / args.agent_test_episodes, step_cum / args.agent_test_episodes


def train_dqn(logger):
    env = EnvWrapper(args.env_name)
    dqn_agent = DQNAgent(args.input_frame_width, args.input_frame_height, env.action_space, args.mini_batch_size,
                         args.replay_buffer_size, args.replay_start_size, args.skip_k_frame,
                         args.learning_rate, args.step_c, args.agent_saving_period, args.gamma, args.training_episodes,
                         args.phi_channel, args.epsilon_max, args.epsilon_min,
                         args.exploration_steps, args.device)
    epoch_i = 0
    episode_in_epoch = 1
    training_steps = 0
    log_reward = 0
    for episode_i in range(args.training_episodes):
        state, _ = env.reset()
        done = False
        reward_raw = 0
        truncated = False
        inf = ''
        step_i = 0
        frame_num = 0
        reward_episode = 0
        while (not done) and (not truncated):
            obs = dqn_agent.perception_mapping(state, step_i)
            reward = dqn_agent.reward_shaping(reward_raw, step_i)
            if len(dqn_agent.memory) > args.replay_start_size and step_i > args.no_op:
                action = dqn_agent.select_action(obs)
            else:
                action = dqn_agent.select_action(obs, RandomAction())
            dqn_agent.store(obs, action, reward, done, truncated, inf)
            dqn_agent.train_step(step_i)
            next_state, reward_raw, done, truncated, inf = env.step(action)
            reward_episode += reward_raw
            state = next_state
            training_steps += 1
            frame_num += 1
            if training_steps % args.batch_num_per_epoch*args.skip_k_frame == 0:
                # run_test = True
                epoch_i += 1
            step_i += 1
        dqn_agent.store_termination()
        # if run_test:
            # avg_reward, avg_steps = test(dqn_agent, args.agent_test_episodes)
        logger(f'agent {epoch_i}: reward episode : {reward_episode}')
        logger(f'agent {epoch_i}: epsilon: {dqn_agent.exploration_method.epsilon}')
        logger(f'agent {epoch_i}: frames episode : {frame_num}')
        episode_in_epoch += 1


if __name__ == '__main__':
    logger_ = Logger(args.env_name, args.log_path)
    train_dqn(logger_)
