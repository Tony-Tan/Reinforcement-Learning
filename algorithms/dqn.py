import argparse
from agents.dqn_agent import *
from environments.env_wrapper import EnvWrapper
import copy
from exploration.epsilon_greedy import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='ALE/Pong-v5', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--mini_batch_size', default=32, type=int,
                    help='ccn training batch size，default: 32')
parser.add_argument('--batch_num_per_epoch', default=5000, type=int,
                    help='each epoch contains how many updates，default: 32')
parser.add_argument('--replay_buffer_size', default=100000, type=int,
                    help='memory buffer size ，default: 15000')
parser.add_argument('--training_episodes', default=100000, type=int,
                    help='max training episodes，default: 100000')
parser.add_argument('--skip_k_frame', default=4, type=int,
                    help='dqn skip k frames each step，default: 4')
parser.add_argument('--phi_channel', default=4, type=int,
                    help='phi temp size, default: 4')
parser.add_argument('--device', default='cuda', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--input_frame_width', default=84, type=int,
                    help='cnn input image width, default: 84')
parser.add_argument('--input_frame_height', default=110, type=int,
                    help='cnn input image height，default: 110')
parser.add_argument('--min_sample_size_before_update', default=200, type=int,
                    help='min data size before training cnn ，default: 2000')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='value decay, default: 0.99')
parser.add_argument('--save_path', default='./data_log/', type=str,
                    help='model save path ，default: ./model/')
parser.add_argument('--log_path', default='../exps/dqn/', type=str,
                    help='log save path，default: ./log/')
parser.add_argument('--learning_rate', default=1e-5, type=float,
                    help='cnn learning rate，default: 1e-5')
parser.add_argument('--step_c', default=100, type=int,
                    help='synchronise target value network periods，default: 100')
parser.add_argument('--epsilon_min', default=0.1, type=float,
                    help='min epsilon of epsilon-greedy，default: 0.1')
parser.add_argument('--epsilon_for_test', default=0.01, type=float,
                    help='epsilon of epsilon-greedy for testing agent，default: 0.05')
parser.add_argument('--agent_test_period', default=100, type=int,
                    help='agent test period(episode)，default: 100')
parser.add_argument('--agent_test_episodes', default=50, type=int,
                    help='agent test episode，default: 100')
parser.add_argument('--agent_saving_period', default=80000, type=int,
                    help='agent saving period(episode)，default: 80000')
args = parser.parse_args()


def test(agent: DQNAgent):
    env = EnvWrapper(args.env_name, logger_)
    exploration_method = EpsilonGreedy(args.epsilon_for_test)
    reward_cum = 0
    for i in range(args.agent_test_episodes):
        state, _ = env.reset()
        done = False
        step_i = 0
        while not done:
            obs = agent.perception_mapping(state, step_i)
            action = agent.select_action(obs, exploration_method)
            next_state, reward, done, truncated, inf = env.step(action)
            reward_cum += reward
            state = next_state
            step_i += 1
    logger_(f'agent test: average reward of an episode: {reward_cum/args.agent_test_episodes}')


def train_dqn():
    env = EnvWrapper(args.env_name, logger_)
    dqn_agent = DQNAgent(args.input_frame_width, args.input_frame_height, env.action_space, args.mini_batch_size,
                         args.replay_buffer_size, args.min_sample_size_before_update, args.skip_k_frame,
                         args.learning_rate, args.step_c, args.agent_saving_period, args.gamma, args.training_episodes,
                         args.phi_channel, args.device)
    epoch_i = 0
    episode_i = 0
    for training_group_i in range(int(args.training_episodes/args.agent_test_period)):
        for i in tqdm(range(0, args.agent_test_period), desc=f'training dqn episode {episode_i} '
                                                             f'to {episode_i + args.agent_test_period}'):
            state, _ = env.reset()
            done = False
            reward_raw = 0
            truncated = False
            inf = ''
            step_i = 0
            while not done:
                obs = dqn_agent.perception_mapping(state, step_i)
                reward = dqn_agent.reward_shaping(reward_raw, step_i)
                action = dqn_agent.select_action(obs)
                dqn_agent.store(obs, action, reward, done, truncated, inf)
                dqn_agent.train_step(step_i)
                next_state, reward_raw, done, truncated, inf = env.step(action)
                state = next_state
                step_i += 1
                if step_i % args.batch_num_per_epoch == 0:
                    epoch_i += 1
            episode_i += 1
        logger_(f'agent train: replay buffer current length: {len(dqn_agent.memory)}')
        test(dqn_agent)


if __name__ == '__main__':
    logger_ = Logger(args.log_path)
    train_dqn()
