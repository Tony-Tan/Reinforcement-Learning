import copy
import torch
from core.rl_elements import *
from DDPG.ddpg import DDPG_exp
import gym
import argparse


parser = argparse.ArgumentParser(description='PyTorch DDPG algorithm for continuous control environment')
parser.add_argument('--env_name', default='InvertedDoublePendulum-v2', type=str,
                    help='Mujoco Gym environment (default: InvertedDoublePendulum-v2)')
parser.add_argument('--critic_hidden_layer', default=[400, 300], nargs='+', type=int,
                    help='critic hidden perceptron size')
parser.add_argument('--actor_hidden_layer', default=[400, 300], nargs='+', type=int,
                    help='acot hidden perceptron size')
parser.add_argument('--critic_lr', default=1e-3, type=float,
                    help='critic learning rate')
parser.add_argument('--actor_lr', default=1e-3, type=float,
                    help='actor learning rate')
parser.add_argument('--batch_size', default=100, type=int,
                    help='actor critic update batch_size')
parser.add_argument('--discounted_rate', default=0.99, type=float,
                    help='episodic return discounted rate $\\gamma$ in the paper')
parser.add_argument('--buffer_size', default=1000000, help='buffer size')
parser.add_argument('--steps_prepare_randomly', default=10000, type=int,
                    help='random generated steps before first update')
parser.add_argument('--steps_prepare', default=1000, type=int,
                    help='steps generated by actor before first update')
parser.add_argument('--polyak', default=.995, type=float,
                    help='polyak coefficient')
parser.add_argument('--round_num', default=3000000, type=int,
                    help='how many round to play the game')
parser.add_argument('--actor_update_period', default=2, type=int,
                    help='update actor between critics update')
parser.add_argument('--recording_period', default=2000, type=int,
                    help='record checkpoints period')
parser.add_argument('--action_noise', default=0.2, type=float,
                    help='action noise to estimate q(obs2,action)')
parser.add_argument('--action_noise_range', default=[-0.5, 0.5], nargs='+', type=float,
                    help='range of action noise to estimate q(obs2,action)')
parser.add_argument('--trajectory_size_each_round', default=50, type=int,
                    help='how many steps to play between each update')
parser.add_argument('--training_device', default='cpu', type=str,
                    help='training mlp on what device, could be mps, cpu, cuda')
parser.add_argument('--test_device', default='cpu', type=str,
                    help='test agent on what device, could be mps, cpu, cuda')
parser.add_argument('--agent_path', default='./data/models/checkpoint.pt', type=str,
                    help='agent models, actors, critics save path')
parser.add_argument('--experiment_log_path', default='./data/log', type=str,
                    help='experiment save path')
args = parser.parse_args()
print(args)


class TD3_Agent(Agent):
    def __init__(self, observation_space, action_space, actor_mlp_hidden_layer,
                 critic_mlp_hidden_layer, path='./data/models'):
        super(TD3_Agent, self).__init__('TD3', path)
        # initialize neural networks
        self.actor = MLPGaussianActorManuSTD(observation_space, action_space, actor_mlp_hidden_layer, torch.nn.ReLU,
                                             output_action=torch.nn.Tanh)
        self.actor_tar = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = MLPCritic(observation_space, action_space, critic_mlp_hidden_layer, torch.nn.ReLU)
        self.critic_tar_1 = copy.deepcopy(self.critic)
        self.critic_1_loss = torch.nn.MSELoss()
        self.critic_1_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_2 = MLPCritic(observation_space, action_space, critic_mlp_hidden_layer, torch.nn.ReLU)
        self.critic_tar_2 = copy.deepcopy(self.critic_2)
        self.critic_2_loss = torch.nn.MSELoss()
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=args.critic_lr)

        self.action_min = self.actor.action_low
        self.action_max = self.actor.action_high

    def save(self, epoch: int):
        self.checkpoint = {
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'actor_tar_state_dict': self.actor_tar.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),

            'critic_state_dict': self.critic.state_dict(),
            'critic_tar_1_state_dict': self.critic_tar_1.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),

            'critic_2_state_dict': self.critic_2.state_dict(),
            'critic_tar_2_state_dict': self.critic_tar_2.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict()
            }
        torch.save(self.checkpoint, self.path)

    def load(self):
        if os.path.exists(self.path):
            checkpoint = torch.load(self.path, map_location=torch.device(args.training_device))
            self.start_epoch = checkpoint['epoch']

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_tar.load_state_dict(checkpoint['actor_tar_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_tar_1.load_state_dict(checkpoint['critic_tar_1_state_dict'])
            self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])

            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.critic_tar_2.load_state_dict(checkpoint['critic_tar_2_state_dict'])
            self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
            # self.critic_2_loss = checkpoint['critic_2_loss']
            self.actor.eval()
            self.actor_tar.eval()
            self.critic.eval()
            self.critic_tar_1.eval()
            self.critic_2.eval()
            self.critic_tar_2.eval()

    def update_actor_critic(self, epoch_num: int, data: DataBuffer, update_time: int, device: str,
                            log_writer: SummaryWriter):
        batch_size = args.batch_size
        gamma = args.discounted_rate
        data_sample = data.sample(batch_size * update_time)
        reward_tensor = torch.as_tensor(data_sample['reward'], dtype=torch.float32).to(device)
        termination_tensor = torch.as_tensor(data_sample['termination'], dtype=torch.float32).to(device)
        obs_tensor = torch.as_tensor(data_sample['obs'], dtype=torch.float32).to(device)
        next_obs_tensor = torch.as_tensor(data_sample['next_obs'], dtype=torch.float32).to(device)
        action_tensor = torch.as_tensor(data_sample['action'], dtype=torch.float32).to(device)
        self.actor.to(device)
        self.actor_tar.to(device)
        self.critic.to(device)
        self.critic_tar_1.to(device)
        self.critic_2.to(device)
        self.critic_tar_2.to(device)
        self.action_min = self.action_min.to(device)
        self.action_max = self.action_max.to(device)
        average_residual_1 = 0
        average_residual_2 = 0
        policy_loss_item = 0
        for i in range(update_time):
            start_ptr = i * batch_size
            end_ptr = (i + 1) * batch_size
            # update main networks
            with torch.no_grad():
                new_action_tensor, _ = self.actor_tar(next_obs_tensor[start_ptr:end_ptr])
                action_noise = torch.randn_like(new_action_tensor).to(device) * args.action_noise
                action_noise = torch.clip(action_noise, min=args.action_noise_range[0], max=args.action_noise_range[1])
                new_action_tensor = torch.clip(new_action_tensor+action_noise, min=self.action_min, max=self.action_max)
                critic_1 = self.critic_tar_1(next_obs_tensor[start_ptr:end_ptr], new_action_tensor)
                critic_2 = self.critic_tar_2(next_obs_tensor[start_ptr:end_ptr], new_action_tensor)
                min_critic_value = torch.min(critic_1, critic_2)
                value_target = reward_tensor[start_ptr:end_ptr] + gamma * \
                    (1 - termination_tensor[start_ptr:end_ptr]) * min_critic_value
            value_target.require_grad = False
            # update critic 1
            critic_output = self.critic(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_1_loss(value_target, critic_output)
            self.critic_1_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_1_optimizer.step()
            average_residual_1 += critic_loss.item()

            # update critic 2
            critic_output = self.critic_2(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_2_loss(value_target, critic_output)
            self.critic_2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_2_optimizer.step()
            average_residual_2 += critic_loss.item()

            if i % args.actor_update_period == 0:
                actor_output, _ = self.actor(obs_tensor[start_ptr:end_ptr])
                for p in self.critic.parameters():
                    p.requires_grad = False
                q_value = self.critic(obs_tensor[start_ptr:end_ptr], actor_output)
                average_q_value = - q_value.mean()
                self.actor_optimizer.zero_grad()
                average_q_value.backward()
                self.actor_optimizer.step()
                policy_loss_item += average_q_value.item()
                for p in self.critic.parameters():
                    p.requires_grad = True
                # update target
                polyak_average(self.actor_tar, self.actor, args.polyak, self.actor_tar)
                polyak_average(self.critic_tar_1, self.critic, args.polyak, self.critic_tar_1)
                polyak_average(self.critic_tar_2, self.critic_2, args.polyak, self.critic_tar_2)

        if epoch_num % 200 == 0:
            average_residual_1 /= update_time
            average_residual_2 /= update_time
            print_time()
            print('\t\t regression state value for advantage; epoch: ' + str(epoch_num))
            print('\t\t value loss for critic_1: ' + str(average_residual_1))
            print('\t\t value loss for critic_2: ' + str(average_residual_2))
            print('\t\t policy loss: ' + str(policy_loss_item))
            print('-----------------------------------------------------------------')
            log_writer.add_scalars('loss/value_loss', {'q_1': average_residual_1,
                                                       'q_2': average_residual_2}, epoch_num)
            log_writer.add_scalar('loss/policy_loss', policy_loss_item, epoch_num)


class TD3_exp(DDPG_exp):
    def __init__(self, env, agent, buffer_size, discounted_rate, normalize=False,
                 log_path='./data/log/', env_data_path='./data/models/'):
        super(TD3_exp, self).__init__(env,  agent, buffer_size, discounted_rate,
                                      normalize=normalize, log_path=log_path,
                                      env_data_path=env_data_path)


if __name__ == '__main__':
    env_ = gym.make(args.env_name)
    agent_ = TD3_Agent(env_.observation_space, env_.action_space,
                       args.critic_hidden_layer, args.actor_hidden_layer,path=args.agent_path)
    agent_.load()
    experiment = TD3_exp(env_, agent_, args.buffer_size, args.discounted_rate,
                         False, args.experiment_log_path,
                         args.agent_path)
    experiment.play(args.round_num, args.trajectory_size_each_round, args.training_device,
                    args.test_device, args.recording_period)
    env_.close()
