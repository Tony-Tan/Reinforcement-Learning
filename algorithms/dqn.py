import argparse
from agents.dqn_agent import *
from environments.env_wrapper import EnvWrapper
from exploration.epsilon_greedy import *
import copy
import cv2
from tqdm import tqdm
from multiprocessing import Process, Queue, set_start_method
from utils.hyperparameters import Hyperparameters

parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='ALE/Pong-v5', type=str,
                    help='openai gym environment (default: ALE/Spaceinvaders-v5)')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--save_path', default='./data_log/', type=str,
                    help='model save path ，default: ./model/')
parser.add_argument('--log_path', default='../exps/dqn/', type=str,
                    help='log save path，default: ./log/')
parser.add_argument('--agent_saving_period', default=80000, type=int,
                    help='agent saving period(episode)，default: 80000')

cfg = Hyperparameters(parser, '../configs/dqn.yaml')
cfg.print()


def test(agent: DQNAgent, test_episodes: int):
    env = EnvWrapper(cfg['env_name'])
    exploration_method = EpsilonGreedy(cfg['epsilon_for_test'])
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
    return reward_cum / cfg['agent_test_episodes'], step_cum / cfg['agent_test_episodes']


def train_dqn(logger):
    """
    train dqn agent
    :param logger:  logger
    :return:
    """
    env = EnvWrapper(cfg['env_name'])

    dqn_agent = DQNAgent(cfg['input_frame_width'], cfg['input_frame_height'], env.action_space, cfg['mini_batch_size'],
                         cfg['replay_buffer_size'], cfg['replay_start_size'], cfg['skip_k_frame'],
                         cfg['learning_rate'], cfg['step_c'], cfg['agent_saving_period'], cfg['gamma'],
                         cfg['training_steps'], cfg['phi_channel'], cfg['epsilon_max'], cfg['epsilon_min'],
                         cfg['exploration_steps'], cfg['device'], logger)

    epoch_i = 0
    training_steps = 0
    while training_steps < cfg['training_steps']:
        state, _ = env.reset()
        done = False
        reward_raw = 0
        truncated = False
        inf = ''
        step_i = 0
        run_test = False
        while (not done) and (not truncated):
            obs = dqn_agent.perception_mapping(state, step_i)
            reward = dqn_agent.reward_shaping(reward_raw, step_i)
            if len(dqn_agent.memory) > cfg['replay_start_size'] and step_i > cfg['no_op']:
                action = dqn_agent.select_action(obs)
            else:
                action = dqn_agent.select_action(obs, RandomAction())
            dqn_agent.store(obs, action, reward, done, truncated, inf)
            dqn_agent.train_step(step_i)
            next_state, reward_raw, done, truncated, inf = env.step(action)
            state = next_state
            if training_steps % cfg['batch_num_per_epoch']*cfg['skip_k_frame'] == 0:
                run_test = True
                epoch_i += 1
            training_steps += 1
            step_i += 1
        dqn_agent.store_termination()
        if run_test:
            avg_reward, avg_steps = test(dqn_agent, cfg['agent_test_episodes'])
            logger.tb_scalar('avg_reward', avg_reward, epoch_i)
            logger.tb_scalar('avg_steps', avg_steps, epoch_i)
            logger.tb_scalar('epsilon', dqn_agent.exploration_method.epsilon, epoch_i)
            logger.msg(f'{epoch_i} avg_reward: ' + str(avg_reward))
            logger.msg(f'{epoch_i} avg_steps: ' + str(avg_steps))
            logger.msg(f'{epoch_i} epsilon: ' + str(dqn_agent.exploration_method.epsilon))


if __name__ == '__main__':
    logger_ = Logger(cfg['env_name'], cfg['log_path'])
    logger_.msg('\nparameters:' + str(cfg))
    train_dqn(logger_)
