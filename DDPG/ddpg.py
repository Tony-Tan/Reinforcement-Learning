import torch.optim

from core.rl_elements import *
from sklearn.utils import shuffle
import gym
import copy


class MLPCritic(Critic):
    def __init__(self, input_state_size: int, input_action_size: int):
        super(MLPCritic, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(input_state_size + input_action_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), 1)
        value = self.linear_stack(x)
        return value


class GaussianActor(Actor):
    def __init__(self, state_dim, action_dim, std_init):
        super(GaussianActor, self).__init__()
        self.std = std_init * np.ones(action_dim, dtype=np.float32)
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.linear_mlp_stack(x) * 0.4
        return mu

    def distribution(self, x: torch.Tensor) -> (np.ndarray, np.ndarray):
        with torch.no_grad():
            mu = self.forward(x).cpu().numpy()
        return mu, self.std


class DDPG_Agent(Agent):
    def __init__(self, state_dim, action_dim, agent_name='DDPG', path='./data/models'):
        self.hyperparameter = Hyperparameter(path)
        self.hyperparameter['batch_size'] = 100
        self.hyperparameter['actor_main_lr'] = 1e-3
        self.hyperparameter['critic_main_lr'] = 1e-3
        self.hyperparameter['discounted_rate'] = 0.99
        self.hyperparameter['polyak'] = 0.995
        self.hyperparameter.load()
        super(DDPG_Agent, self).__init__(agent_name, path)
        self.actor = GaussianActor(state_dim, action_dim, .1) if (self.actor is None) else self.actor
        self.critic = MLPCritic(state_dim, action_dim) if (self.critic is None) else self.critic
        self.actor_main = copy.deepcopy(self.actor)
        self.critic_main = copy.deepcopy(self.critic)
        self.critic_main_loss = torch.nn.MSELoss()
        self.critic_main_optimizer = torch.optim.SGD(self.critic_main.parameters(),
                                                      lr=self.hyperparameter['critic_main_lr'])
        self.actor_main_optimizer = torch.optim.SGD(self.actor_main.parameters(),
                                                     lr=self.hyperparameter['actor_main_lr'])

    def reaction(self, state: np.ndarray):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            self.actor_main.to('cpu')
            mu, std = self.actor_main.distribution(state_tensor)
            action = np.random.normal(mu, std)
            return action, mu, std

    def polyak_average(self, model1, model2, dist_model):
        beta = self.hyperparameter['polyak']
        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(beta * param1.data + (1 - beta) * dict_params2[name1].data)

        dist_model.load_state_dict(dict_params2)

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
        self.critic_main.to(device)
        i = 0
        average_residual = 0
        while i < update_time:
            start_ptr = i * batch_size
            end_ptr = (i + 1) * batch_size
            # update main networks
            with torch.no_grad():
                new_action_tensor = self.actor(next_obs_tensor[start_ptr:end_ptr])
                value_target = reward_tensor[start_ptr:end_ptr] + \
                    gamma * (1 - termination_tensor[start_ptr:end_ptr]) * \
                    self.critic(next_obs_tensor[start_ptr:end_ptr], new_action_tensor)
            value_target.require_grad = False

            critic_output = self.critic_main(obs_tensor[start_ptr:end_ptr], action_tensor[start_ptr:end_ptr])
            critic_loss = self.critic_main_loss(value_target, critic_output)
            self.critic_main_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_main_optimizer.step()
            average_residual += critic_loss.item()

            actor_output = self.actor_main(obs_tensor[start_ptr:end_ptr])
            for p in self.critic_main.parameters():
                p.requires_grad = False
            q_value = self.critic_main(obs_tensor[start_ptr:end_ptr], actor_output)
            average_q_value = - q_value.mean()
            self.actor_main_optimizer.zero_grad()
            average_q_value.backward()
            self.actor_main_optimizer.step()
            for p in self.critic_main.parameters():
                p.requires_grad = True
            # update target
            self.polyak_average(self.actor, self.actor_main, self.actor)
            self.polyak_average(self.critic, self.critic_main, self.critic)
            i += 1

        if epoch_num % 200 == 0:
            average_residual /= update_time
            print_time()
            print('\t\t regression state value for advantage; epoch: ' + str(epoch_num))
            print('\t\t value loss: ' + str(average_residual))
            print('-----------------------------------------------------------------')
            log_writer.add_scalar('value loss', average_residual, epoch_num)


class DDPG_exp(RLExperiment):
    def __init__(self, env, state_dim, action_dim, agent: DDPG_Agent, buffer_size, log_path=None):
        buffer_template = {'obs': state_dim, 'action': action_dim,
                           'next_obs': state_dim, 'reward': 0, 'termination': 0}
        super(DDPG_exp, self).__init__(env, 0.99, agent, buffer_size, buffer_template, log_path)
        # self.buffer = DataBuffer(buffer_size, data_template)
        # self.trajectory_size = buffer_size
        self.reward_std = None
        self.state_std = None
        self.state_mean = None
        self.env_info('./data/models/')

    def generate_trajectories(self, trajectory_size: int):
        # current_state = self.env.reset()
        if self.buffer['termination'][self.buffer.ptr-1]:
            current_state = self.env.reset()
            current_state = np.float32((current_state - self.state_mean) / self.state_std)
        else:
            current_state = self.buffer['next_obs'][self.buffer.ptr-1]
        # current_state = np.float32((current_state - self.state_mean) / self.state_std)
        for _ in range(trajectory_size):
            # current_state = np.float32(current_state)
            action, _, _ = self.agent.reaction(current_state)
            new_state, reward, is_done, _ = self.env.step(action)
            new_state = np.float32((new_state - self.state_mean) / self.state_std)
            self.buffer.push([current_state, action, new_state, reward / self.reward_std, is_done])
            # new_state = np.float32(new_state)
            # self.buffer.push([current_state, action, new_state, reward, is_done])
            if is_done:
                current_state = self.env.reset()
                current_state = np.float32((current_state - self.state_mean) / self.state_std)
            else:
                current_state = new_state

    def fill_buffer(self, rate):
        current_state = self.env.reset()
        current_state = np.float32((current_state - self.state_mean) / self.state_std)
        for _ in range(int(self.buffer.max_size*rate)):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            new_state = np.float32((new_state - self.state_mean) / self.state_std)
            self.buffer.push([current_state, action, new_state, reward / self.reward_std, is_done])
            if is_done:
                current_state = self.env.reset()
                current_state = np.float32((current_state - self.state_mean) / self.state_std)
            else:
                current_state = new_state

    def env_info(self, data_path):
        if os.path.exists(os.path.join(data_path, 'state_mean.npy')):
            state_mean_path = os.path.join(data_path, 'state_mean.npy')
            self.state_mean = np.load(state_mean_path)
            state_std_path = os.path.join(data_path, 'state_std.npy')
            self.state_std = np.load(state_std_path)
            reward_std_path = os.path.join(data_path, 'reward_std.npy')
            self.reward_std = np.load(reward_std_path)
            print('data loaded .... ')
            print('state mean', self.state_mean)
            print('state std', self.state_std)
            print('reward std', self.reward_std)
        else:
            state_list = []
            discount_return_list = []
            for i in range(1000):
                state = self.env.reset()
                state_list.append(state)
                total_reward = 0
                step_i = 0
                while True:
                    random_action = self.env.action_space.sample()
                    new_state, reward, is_done, _ = self.env.step(random_action)
                    state_list.append(new_state)
                    total_reward += np.power(self.gamma, step_i) * reward
                    step_i += 1
                    if is_done:
                        break
                discount_return_list.append(total_reward)
            self.env.close()
            reward_np = np.array(discount_return_list, dtype=np.float32)
            state_np = np.array(state_list, dtype=np.float32)
            self.reward_std = np.std(reward_np) + 1e-5  # not be zero
            self.state_std = np.std(state_np, axis=0) + 1e-5  # not be zero
            self.state_mean = np.mean(state_np, axis=0)
            state_mean_path = os.path.join(data_path, 'state_mean.npy')
            np.save(state_mean_path, self.state_mean)
            state_std_path = os.path.join(data_path, 'state_std.npy')
            np.save(state_std_path, self.state_std)
            reward_std_path = os.path.join(data_path, 'reward_std.npy')
            np.save(reward_std_path, self.reward_std)
            print('state reward info saved!')

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
                action = self.agent.actor(obs_tensor).detach().numpy()
                obs, reward, done, _ = env.step(action)
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

    def play(self, round_num=1000000, trajectory_size_each_round=100):
        start_round_num = self.agent.start_epoch
        print_time()
        self.fill_buffer(0.01)
        self.generate_trajectories(int(round_num*0.1))
        for round_i in range(start_round_num, round_num):
            self.generate_trajectories(trajectory_size_each_round)
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            self.agent.update_actor_critic(round_i, self.buffer, 100, 'cpu', self.exp_log_writer)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

            if round_i % 1000 == 0:
                self.test(round_i, 100, 'cpu')
                self.agent.save(round_i)
                self.exp_log_writer.add_scalars('lr', {'value': self.agent.hyperparameter['critic_main_lr'],
                                                       'policy': self.agent.hyperparameter['actor_main_lr']}, round_i)


if __name__ == '__main__':
    env_ = gym.make('HumanoidStandup-v2')
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    agent = DDPG_Agent(state_dim_, action_dim_)
    experiment = DDPG_exp(env_, state_dim_, action_dim_, agent, 1000000, './data/log/')
    experiment.play(round_num=1000000, trajectory_size_each_round=100)
    env_.close()
