import argparse
from agents.dueling_dqn_agent import *
from environments.env_wrapper import EnvWrapper
from exploration.epsilon_greedy import *
from tools.dqn_play_ground import DQNPlayGround
from utils.hyperparameters import Hyperparameters

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='PyTorch Dueling DQN training arguments')
parser.add_argument('--env_name', default='ALE/Atlantis-v5', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='calculation device default: cuda')
parser.add_argument('--log_path', default='../exps/dueling_dqn/', type=str,
                    help='log save path，default: ./log/')

# Load hyperparameters from yaml file
cfg = Hyperparameters(parser, '../configs/dqn.yaml')



def main():
    logger = Logger(cfg['env_name'], cfg['log_path'])
    logger.msg('\nparameters:' + str(cfg))
    env = EnvWrapper(cfg['env_name'], repeat_action_probability=0, frameskip=cfg['skip_k_frame'])
    dueling_dqn_agent = DuelingDQNAgent(cfg['input_frame_width'], cfg['input_frame_height'], env.action_space, cfg['mini_batch_size'],
                         cfg['replay_buffer_size'], cfg['replay_start_size'], cfg['learning_rate'], cfg['step_c'],
                         cfg['agent_saving_period'], cfg['gamma'], cfg['training_steps'], cfg['phi_channel'],
                         cfg['epsilon_max'], cfg['epsilon_min'], cfg['exploration_steps'], cfg['device'], logger)
    dqn_pg = DQNPlayGround(dueling_dqn_agent, env, cfg, logger)
    dqn_pg.train()


if __name__ == '__main__':
    main()