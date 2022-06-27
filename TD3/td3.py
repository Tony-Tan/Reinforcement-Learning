import copy

import torch

from DDPG.ddpg import *


class TD3_Agent(DDPG_Agent):
    def __init__(self, state_dim, action_dim,min_action, max_action):
        self.critic_2 = None
        super(TD3_Agent, self).__init__(state_dim, action_dim, 'TD3')
        self.hyperparameter['actor_update_period'] = 2
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

    def save(self, epoch_num):
        super(TD3_Agent, self).save(epoch_num)
        value_module_path = os.path.join(self.path, self.model_name_) + '_critic_2.pt'
        torch.save(self.critic_2, value_module_path)

    def load(self, model_name=None):
        super(TD3_Agent, self).load(model_name)
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
        self.actor_main.to(device)
        self.critic_1.to(device)
        self.critic_main_1.to(device)
        self.critic_2.to(device)
        self.critic_main_2.to(device)
        i = 0
        average_residual_1 = 0
        average_residual_2 = 0
        while i < update_time:
            start_ptr = i * batch_size
            end_ptr = (i + 1) * batch_size
            # update main networks
            with torch.no_grad():
                new_action_tensor = self.actor(next_obs_tensor[start_ptr:end_ptr]) + \
                                    torch.as_tensor(np.random.normal(0, 0.1, [batch_size, self.action_dim]),
                                                    dtype=torch.float32).to(device)
                new_action_tensor = torch.clip(new_action_tensor, min=self.min_action, max=self.max_action)
                value_target = reward_tensor[start_ptr:end_ptr] + \
                    gamma * (1 - termination_tensor[start_ptr:end_ptr]) * \
                    torch.min(self.critic_1(next_obs_tensor[start_ptr:end_ptr], new_action_tensor),
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
                actor_output = self.actor_main(obs_tensor[start_ptr:end_ptr])
                for p in self.critic_main.parameters():
                    p.requires_grad = False
                q_value = self.critic_main_1(obs_tensor[start_ptr:end_ptr], actor_output)
                average_q_value = - q_value.mean()
                self.actor_main_optimizer.zero_grad()
                average_q_value.backward()
                self.actor_main_optimizer.step()
                for p in self.critic_main.parameters():
                    p.requires_grad = True
                # update target
                self.polyak_average(self.actor, self.actor_main, self.actor)
                self.polyak_average(self.critic_1, self.critic_main_1, self.critic_1)
                self.polyak_average(self.critic_2, self.critic_main_2, self.critic_2)
            i += 1

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
    def __init__(self, env, state_dim, action_dim, agent: DDPG_Agent, buffer_size, log_path=None):
        super(TD3_exp, self).__init__(env, state_dim, action_dim, agent, buffer_size, log_path)


if __name__ == '__main__':
    env_ = gym.make('HumanoidStandup-v2')
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    min_action_ = -0.4
    max_action_ = 0.4
    agent_ = TD3_Agent(state_dim_, action_dim_, min_action_, max_action_)
    experiment = TD3_exp(env_, state_dim_, action_dim_, agent_, 1000000, './data/log/')
    experiment.play(round_num=1000000, trajectory_size_each_round=100)
    env_.close()
