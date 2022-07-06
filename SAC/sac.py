import copy

import torch

from DDPG.ddpg import *


class SAC_Agent(DDPG_Agent):
    def __init__(self, state_dim, action_dim, actor_mlp_hidden_layer, critic_mlp_hidden_layer,
                 action_min, action_max, path='./data/models'):
        super(SAC_Agent, self).__init__(state_dim, action_dim, actor_mlp_hidden_layer, critic_mlp_hidden_layer,
                                        action_min, action_max, 'SAC', path=path)
        self.hyperparameter['heat'] = 0.2
        self.hyperparameter['actor_main_lr'] = 1e-3
        self.hyperparameter['critic_main_lr'] = 1e-3
        if isinstance(self.actor, MLPGaussianActorManuSTD):
            self.actor = MLPGaussianActorSquashing(state_dim, action_dim, actor_mlp_hidden_layer,
                                                   hidden_action_fc=torch.nn.ReLU)
            self.critic = MLPCritic(state_dim, action_dim, critic_mlp_hidden_layer, torch.nn.ReLU)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.hyperparameter['actor_main_lr'])

        self.critic_tar = copy.deepcopy(self.critic)
        self.critic_loss = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.hyperparameter['critic_main_lr'])

        self.critic_2 = MLPCritic(state_dim, action_dim, critic_mlp_hidden_layer, torch.nn.ReLU)
        self.critic_2_loss = torch.nn.MSELoss()
        self.critic_tar_2 = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=self.hyperparameter['critic_main_lr'])

    # def reaction(self, state: np.ndarray, device='cpu'):
    #     with torch.no_grad():
    #         state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
    #         self.actor.to(device)
    #         action = self.actor.act(state_tensor)
    #         return action.cpu().numpy()

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
        self.critic.to(device)
        self.critic_tar.to(device)
        self.critic_2.to(device)
        self.critic_tar_2.to(device)
        average_residual_1 = 0
        average_residual_2 = 0
        policy_loss_item = 0
        for i in range(update_time):
            start_ptr = i * batch_size
            end_ptr = (i + 1) * batch_size
            # update main networks
            with torch.no_grad():
                action, log_pro = self.actor.act_logpro(next_obs_tensor[start_ptr:end_ptr])
                entropy = -log_pro
                q_value_1 = self.critic_tar(next_obs_tensor[start_ptr:end_ptr], action)
                q_value_2 = self.critic_tar_2(next_obs_tensor[start_ptr:end_ptr], action)
                min_q_value = torch.min(q_value_1, q_value_2)
                termination_coe = 1. - termination_tensor[start_ptr:end_ptr]
                value_target = reward_tensor[start_ptr:end_ptr] + \
                    gamma * termination_coe * (min_q_value + self.hyperparameter['heat'] * entropy)

            # update critic 1
            critic_output = self.critic(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_loss(value_target, critic_output)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            average_residual_1 += critic_loss.item()

            # update critic 2
            critic_output = self.critic_2(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_2_loss(value_target, critic_output)
            self.critic_2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_2_optimizer.step()
            average_residual_2 += critic_loss.item()

            # update actor
            actions, action_log_pro = self.actor.act_logpro(obs_tensor[start_ptr:end_ptr])

            for p in self.critic.parameters():
                p.requires_grad = False
            for p in self.critic_2.parameters():
                p.requires_grad = False

            q_value_1 = self.critic(obs_tensor[start_ptr:end_ptr], actions)
            q_value_2 = self.critic_2(obs_tensor[start_ptr:end_ptr], actions)
            q_value = torch.min(q_value_1, q_value_2)

            loss = (self.hyperparameter['heat'] * action_log_pro - q_value).mean()
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            policy_loss_item = loss.item()

            for p in self.critic.parameters():
                p.requires_grad = True
            for p in self.critic_2.parameters():
                p.requires_grad = True
            # update target
            polyak_average(self.critic_tar, self.critic, self.hyperparameter['polyak'], self.critic_tar)
            polyak_average(self.critic_tar_2, self.critic_2, self.hyperparameter['polyak'], self.critic_tar_2)

        if epoch_num % 200 == 0:
            average_residual_1 /= update_time
            average_residual_2 /= update_time
            print_time()
            print('\t\t regression state value for advantage; epoch: ' + str(epoch_num))
            print('\t\t value loss for critic_1: ' + str(average_residual_1))
            print('\t\t value loss for critic_2: ' + str(average_residual_2))
            print('\t\t policy loss : ' + str(policy_loss_item))
            print('-----------------------------------------------------------------')
            log_writer.add_scalars('loss/value_loss', {'q_1': average_residual_1,
                                                       'q_2': average_residual_2}, epoch_num)
            log_writer.add_scalar('loss/policy_loss', policy_loss_item , epoch_num)


class SAC_exp(DDPG_exp):
    def __init__(self, env, state_dim, action_dim, agent: Agent, buffer_size,
                 normalize=True, log_path=None, env_data_path='./data/models/'):
        super(SAC_exp, self).__init__(env, state_dim, action_dim, agent, buffer_size,
                                      normalize=normalize, log_path=log_path, env_data_path=env_data_path)

    def test(self, round_num: int, test_round_num: int, device='cpu'):
        env = copy.deepcopy(self.env)
        total_reward = 0.0
        total_steps = 0
        self.agent.actor.eval()
        self.agent.actor.to(device)
        for i in range(test_round_num):
            obs = env.reset()
            while True:
                if self.normalize:
                    obs = np.float32((obs - self.state_mean) / self.state_std)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)
                with torch.no_grad():
                    action = self.agent.actor.act(obs_tensor, stochastically=False)
                obs, reward, done, _ = env.step(action.cpu().numpy())
                total_reward += reward
                total_steps += 1
                if done:
                    break
        print("Episode done in %.2f steps, total reward %.2f" % (
            total_steps / test_round_num, total_reward / test_round_num))
        env.close()
        self.exp_log_writer.add_scalar('reward', total_reward / test_round_num, round_num)
        self.exp_log_writer.add_scalar('step', total_steps / test_round_num, round_num)
        return total_reward / test_round_num


if __name__ == '__main__':
    env_ = gym.make('InvertedDoublePendulum-v2')
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    min_action_ = -1
    max_action_ = 1
    agent_ = SAC_Agent(state_dim_, action_dim_, [256, 256], [256, 256], action_min=-1, action_max=1)
    experiment = SAC_exp(env_, state_dim_, action_dim_, agent_, 1000000, False, './data/log/')
    experiment.play(round_num=1000000, trajectory_size_each_round=100, training_device='cpu',
                    test_device='cpu', recording_period=2000)
    env_.close()
