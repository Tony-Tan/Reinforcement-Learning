import copy
import torch
from core.rl_elements import *
from DDPG.ddpg import DDPG_exp
import gym

class TD3_Agent(Agent):
    def __init__(self, state_dim, action_dim, actor_mlp_hidden_layer, critic_mlp_hidden_layer,
                 action_min, action_max, path='./data/models'):
        super(TD3_Agent, self).__init__('TD3', path)
        self.hyperparameter['batch_size'] = 100
        self.hyperparameter['actor_main_lr'] = 1e-3
        self.hyperparameter['critic_main_lr'] = 1e-3
        self.hyperparameter['discounted_rate'] = 0.99
        self.hyperparameter['polyak'] = 0.995
        self.hyperparameter['actor_update_period'] = 2
        self.hyperparameter.load()
        self.actor = MLPGaussianActorManuSTD(state_dim, action_dim, actor_mlp_hidden_layer, torch.nn.Tanh,
                                             output_action=torch.nn.Tanh, std_init=.1, std_decay=1.,
                                             mu_output_shrink=action_max)
        self.critic = MLPCritic(state_dim, action_dim, critic_mlp_hidden_layer, torch.nn.Tanh)
        self.critic_2 = MLPCritic(state_dim, action_dim, critic_mlp_hidden_layer, torch.nn.Tanh)
        self.load()
        self.critic_main_1 = copy.deepcopy(self.critic)
        self.critic_main_2 = copy.deepcopy(self.critic_2)
        self.actor_main = copy.deepcopy(self.actor)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.critic_main_1_loss = torch.nn.MSELoss()
        self.critic_main_2_loss = torch.nn.MSELoss()
        self.actor_main_optimizer = torch.optim.Adam(self.actor_main.parameters(),
                                                     lr=self.hyperparameter['actor_main_lr'])
        self.critic_main_1_optimizer = torch.optim.Adam(self.critic_main_1.parameters(),
                                                        lr=self.hyperparameter['critic_main_lr'])
        self.critic_main_2_optimizer = torch.optim.Adam(self.critic_main_2.parameters(),
                                                        lr=self.hyperparameter['critic_main_lr'])

    def save(self, epoch_num):
        super(TD3_Agent, self).save(epoch_num)
        value_module_path = os.path.join(self.path, self.model_name_) + '_critic_2.pt'
        torch.save(self.critic_2, value_module_path)

    def load(self, model_name=None):
        super(TD3_Agent, self).load(model_name)
        if self.model_name_ is not None:
            critic_module_path = os.path.join(self.path, self.model_name_) + '_critic_2.pt'
            self.critic_2 = torch.load(critic_module_path, map_location=torch.device('cpu'))

    def reaction(self, state: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            self.actor_main.to('cpu')
            mu, std = self.actor_main(state_tensor)
            mu = mu.cpu().numpy()
            action = np.random.normal(mu, std)
            return action, mu, std

    def update_actor_critic(self, epoch_num: int, data: DataBuffer, update_time: int, device: str,
                            log_writer: SummaryWriter):
        batch_size = self.hyperparameter['batch_size']
        gamma = self.hyperparameter['discounted_rate']
        data_sample = data.sample(batch_size * update_time)
        reward_tensor = torch.as_tensor(data_sample['reward'], dtype=torch.float32).to(device)
        termination_tensor = torch.as_tensor(data_sample['termination'], dtype=torch.float32).to(device)
        obs_tensor = torch.as_tensor(data_sample['obs'], dtype=torch.float32).to(device)
        next_obs_tensor = torch.as_tensor(data_sample['next_obs'], dtype=torch.float32).to(device)
        action_tensor = torch.as_tensor(data_sample['action'], dtype=torch.float32).to(device)

        self.actor.to(device)
        self.actor_main.to(device)
        self.critic.to(device)
        self.critic_main_1.to(device)
        self.critic_2.to(device)
        self.critic_main_2.to(device)
        i = 0
        average_residual_1 = 0
        average_residual_2 = 0
        for i in range(update_time):
            start_ptr = i * batch_size
            end_ptr = (i + 1) * batch_size
            # update main networks
            with torch.no_grad():
                new_action_tensor = self.actor(next_obs_tensor[start_ptr:end_ptr])[0] + \
                                    torch.as_tensor(np.random.normal(0, 0.1, [batch_size, self.action_dim]),
                                                    dtype=torch.float32).to(device)
                new_action_tensor = torch.clip(new_action_tensor, min=self.action_min, max=self.action_max)
                value_target = reward_tensor[start_ptr:end_ptr] + \
                    gamma * (1 - termination_tensor[start_ptr:end_ptr]) * \
                    torch.min(self.critic(next_obs_tensor[start_ptr:end_ptr], new_action_tensor),
                              self.critic_2(next_obs_tensor[start_ptr:end_ptr], new_action_tensor))
            value_target.require_grad = False
            # update critic 1
            critic_output = self.critic_main_1(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_main_1_loss(value_target, critic_output)
            self.critic_main_1_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_main_1_optimizer.step()
            average_residual_1 += critic_loss.item()

            # update critic 2
            critic_output = self.critic_main_2(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_main_2_loss(value_target, critic_output)
            self.critic_main_2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_main_2_optimizer.step()
            average_residual_2 += critic_loss.item()

            if i % self.hyperparameter['actor_update_period'] == 0:
                actor_output,_ = self.actor_main(obs_tensor[start_ptr:end_ptr])
                for p in self.critic_main_1.parameters():
                    p.requires_grad = False
                q_value = self.critic_main_1(obs_tensor[start_ptr:end_ptr], actor_output)
                average_q_value = - q_value.mean()
                self.actor_main_optimizer.zero_grad()
                average_q_value.backward()
                self.actor_main_optimizer.step()
                for p in self.critic_main_1.parameters():
                    p.requires_grad = True
                # update target
                polyak_average(self.actor, self.actor_main, self.hyperparameter['polyak'], self.actor)
                polyak_average(self.critic, self.critic_main_1, self.hyperparameter['polyak'], self.critic)
                polyak_average(self.critic_2, self.critic_main_2, self.hyperparameter['polyak'], self.critic_2)

        if epoch_num % 200 == 0:
            average_residual_1 /= update_time
            average_residual_2 /= update_time
            print_time()
            print('\t\t regression state value for advantage; epoch: ' + str(epoch_num))
            print('\t\t value loss for critic_1: ' + str(average_residual_1))
            print('\t\t value loss for critic_2: ' + str(average_residual_2))
            print('-----------------------------------------------------------------')
            log_writer.add_scalars('value loss', {'q_1': average_residual_1,
                                                  'q_2': average_residual_2}, epoch_num)


class TD3_exp(DDPG_exp):
    def __init__(self, env, state_dim, action_dim, agent, buffer_size, log_path):
        super(TD3_exp, self).__init__(env, state_dim, action_dim, agent, buffer_size, log_path)


if __name__ == '__main__':
    env_ = gym.make('Swimmer-v3')
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    min_action_ = -1.
    max_action_ = 1.
    agent_ = TD3_Agent(state_dim_, action_dim_, [64, 64], [64, 64], min_action_, max_action_)
    experiment = TD3_exp(env_, state_dim_, action_dim_, agent_, 1000000, './data/log/')
    experiment.play(round_num=1000000, trajectory_size_each_round=100, training_device='cpu',
                    test_device='cpu', recording_period=2000)
    env_.close()
