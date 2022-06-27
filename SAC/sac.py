import copy

import torch

from DDPG.ddpg import *


class StochasticGaussianActor(Actor):
    def __init__(self, state_dim, action_dim):
        super(StochasticGaussianActor, self).__init__()
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh())
        self.mean_fc = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh())
        self.std_fc = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        fea = self.linear_mlp_stack(x)
        mu = self.mean_fc(fea)
        std = self.std_fc(fea) * 0.5
        # return torch.cat([mu, std])
        return mu, std

    # def distribution(self, x: torch.Tensor) -> (np.ndarray, np.ndarray):
    #     with torch.no_grad():
    #         mu = self.forward(x).cpu().numpy()
    #     return mu, self.std

#
# def Gaussian_dist(action,mu,std):
#     pass


class SAC_Agent(DDPG_Agent):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        self.critic_2 = None
        self.actor = None
        super(SAC_Agent, self).__init__(state_dim, action_dim, 'SAC_Agent')
        self.hyperparameter['actor_update_period'] = 2
        self.hyperparameter['heat'] = .2
        self.actor = StochasticGaussianActor(state_dim, action_dim) if \
            (isinstance(self.actor, GaussianActor) or (self.actor is None)) else self.actor
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_action = min_action
        self.max_action = max_action
        self.critic_main_1_loss = torch.nn.MSELoss()
        self.critic_main_2_loss = torch.nn.MSELoss()
        self.critic_1 = self.critic
        self.critic_2 = MLPCritic(state_dim, action_dim) if (self.critic_2 is None) else self.critic_2

        self.critic_main_1 = self.critic_main
        self.critic_main_2 = copy.deepcopy(self.critic_2)

        self.critic_main_1_optimizer = self.critic_main_optimizer
        self.critic_main_2_optimizer = torch.optim.SGD(self.critic_main_2.parameters(),
                                                       lr=self.hyperparameter['critic_main_lr'])
        self.actor_optimizer = torch.optim.SGD(self.actor.parameters(),
                                               lr=self.hyperparameter['actor_main_lr'])

    def reaction(self, state: np.ndarray):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            mu, std = self.actor(state_tensor)
            mu = mu.cpu().numpy()
            std = std.cpu().numpy()
            action = np.random.normal(mu, std)
            return action, mu, std

    def save(self, epoch_num):
        super(SAC_Agent, self).save(epoch_num)
        value_module_path = os.path.join(self.path, self.model_name_) + '_critic_2.pt'
        torch.save(self.critic_2, value_module_path)

    def load(self, model_name=None):
        super(SAC_Agent, self).load(model_name)
        if self.model_name_ is not None:
            critic_module_path = os.path.join(self.path, self.model_name_) + '_critic_2.pt'
            self.critic_2 = torch.load(critic_module_path, map_location=torch.device('cpu'))

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
        # self.actor_main.to(device)
        self.critic_1.to(device)
        self.critic_main_1.to(device)
        self.critic_2.to(device)
        self.critic_main_2.to(device)
        i = self.start_epoch
        average_residual_1 = 0
        average_residual_2 = 0
        for i in range(update_time):
            start_ptr = i * batch_size
            end_ptr = (i + 1) * batch_size
            # update main networks
            with torch.no_grad():
                action, mu, std = self.reaction(next_obs_tensor[start_ptr:end_ptr])
                new_action_tensor = torch.as_tensor(action, dtype=torch.float32)
                # a = np.log(std)
                # b = np.log(np.sqrt(2 * np.pi))
                # c = 0.5 * ((action - mu) * (action - mu)) / (std * std)
                g = (1./(std*np.sqrt(2.*np.pi)))*np.exp(-((action-mu)*(action-mu))/(2*std*std))
                # some tricks
                g /= 50.
                entropy = self.hyperparameter['heat'] * (-np.min(np.log(g), 0))
                entropy_tensor = torch.as_tensor(entropy, dtype=torch.float32)
                q_value_1 = self.critic_1(next_obs_tensor[start_ptr:end_ptr], new_action_tensor)
                q_value_2 = self.critic_2(next_obs_tensor[start_ptr:end_ptr], new_action_tensor)
                min_q_value = torch.min(q_value_1, q_value_2)
                termination_coe = 1 - termination_tensor[start_ptr:end_ptr]
                value_target = reward_tensor[start_ptr:end_ptr] + \
                               gamma * termination_coe * (min_q_value + entropy_tensor)

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

            # update actor
            mu, std = self.actor(obs_tensor[start_ptr:end_ptr])
            actions_sample = np.random.normal(mu.detach().cpu().numpy(), std.detach().cpu().numpy())
            actions_sample = torch.as_tensor(actions_sample, dtype=torch.float32)
            with torch.no_grad():
                q_value_1 = self.critic_main_1(obs_tensor[start_ptr:end_ptr], actions_sample)
                q_value_2 = self.critic_main_2(obs_tensor[start_ptr:end_ptr], actions_sample)
                q_value = torch.min(q_value_1, q_value_2)
            g = 1/(std * torch.sqrt(2 * torch.pi * torch.ones_like(std)))*torch.exp(
                -0.5*(actions_sample-mu)*(actions_sample-mu)/(std*std))
            g = g / 10.
            entropy_tensor = torch.min(torch.log(g), torch.zeros_like(g))
            loss = - torch.mean(q_value - self.hyperparameter['heat'] * entropy_tensor)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            # update target
            self.polyak_average(self.critic_1, self.critic_main_1, self.critic_1)
            self.polyak_average(self.critic_2, self.critic_main_2, self.critic_2)

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

class SAC_exp(DDPG_exp):
    def __init__(self, env, state_dim, action_dim, agent: DDPG_Agent, buffer_size, log_path=None):
        super(SAC_exp, self).__init__(env, state_dim, action_dim, agent, buffer_size, log_path)

    def test(self, round_num: int, test_round_num: int, device='cpu'):
        env = copy.deepcopy(self.env)
        total_reward = 0.0
        total_steps = 0
        self.agent.actor.eval()
        self.agent.actor.to(device)
        for i in range(test_round_num):
            obs = env.reset()
            while True:
                obs = np.float32((obs - self.state_mean) / self.state_std)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
                with torch.no_grad():
                    action, _ = self.agent.actor(obs_tensor)
                obs, reward, done, _ = env.step(action.cpu().numpy())
                total_reward += reward
                total_steps += 1
                if done:
                    break
        print("Episode done in %.2f steps, total reward %.2f" % (
            total_steps / test_round_num, total_reward / test_round_num))
        env.close()
        self.exp_log_writer.add_scalars('reward and step', {'reward': total_reward / test_round_num,
                                                            'step': total_steps / test_round_num}, round_num)
        # self.exp_log_writer.add_scalar('reward', total_reward / test_round_num, round_num)
        return total_reward / test_round_num


if __name__ == '__main__':
    env_ = gym.make('InvertedDoublePendulum-v2')
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    min_action_ = -1
    max_action_ = 1
    agent_ = SAC_Agent(state_dim_, action_dim_, min_action_, max_action_)
    experiment = SAC_exp(env_, state_dim_, action_dim_, agent_, 1000000, './data/log/')
    experiment.play(round_num=1000000, trajectory_size_each_round=100)
    env_.close()
