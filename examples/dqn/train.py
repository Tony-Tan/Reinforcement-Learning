import argparse
from agents import DQN
from environments.envwrapper import EnvWrapper

parser = argparse.ArgumentParser(description='PyTorch dqn training arguments')
parser.add_argument('--env_name', default='ALE/Pong-v5', type=str,
                    help='openai gym environment (default: ALE/Pong-v5)')
parser.add_argument('--mini_batch_size', default=32, type=int, help='ccn training batch size，default: 32')
parser.add_argument('--replay_buffer_size', default=15000, type=int, help='memory buffer size ，default: 15000')
parser.add_argument('--max_training_steps', default=100000, type=int, help='max training episodes，default: 100000')
parser.add_argument('--skip_k_frame', default=4, type=int, help='dqn skip k frames each step，default: 4')
parser.add_argument('--phi_temp_size', default=4, type=int, help='phi temp size, default: 4')
parser.add_argument('--device', default='cuda', type=str, help='calculation device default: cuda')
parser.add_argument('--input_frame_width', default=84, type=int, help='cnn input image width, default: 84')
parser.add_argument('--input_frame_height', default=110, type=int, help='cnn input image height，default: 110')
parser.add_argument('--init_data_size', default=2000, type=int, help='min data size before training cnn ，default: 2000')
parser.add_argument('--gamma', default=0.99, type=float, help='value decay, default: 0.99')
parser.add_argument('--save_path', default='./data_log/', type=str, help='model save path ，default: ./model/')
parser.add_argument('--log_path', default='./data_log/', type=str, help='log save path，default: ./log/')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='cnn learning rate，default: 1e-5')
parser.add_argument('--step_c', default=100, type=int,
                    help='synchronise target value network periods，default: 100')
parser.add_argument('--epsilon_min', default=0.1, type=float,
                    help='min epsilon of epsilon-greedy，default: 0.1')
parser.add_argument('--epsilon_for_test', default=0.05, type=float,
                    help='epsilon of epsilon-greedy for testing agent，default: 0.05')
parser.add_argument('--model_saving_period', default=80000, type=int,
                    help='model saving period(step)，default: 80000')
args = parser.parse_args()


if __name__ == '__main__':
    logger_ = Logger(args.log_path)
    env_ = EnvWrapper(args.env_name, logger_)
    agent = DQN(env_, args.phi_temp_size, args.replay_buffer_size,
                args.skip_k_frame, args.mini_batch_size, args.learning_rate,
                args.input_frame_width, args.input_frame_height, args.init_data_size, args.max_training_steps,
                args.gamma, args.step_c, args.model_saving_period, args.device, args.save_path, logger_)
    agent.training()
