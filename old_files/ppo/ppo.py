import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from generalized_advantage_estimation.gae import GAE
import gym
from sklearn.utils import shuffle


def loss_function(pi_new, pi_old, estimate_advantage, epsilon):
    r_t = pi_new / pi_old
    r_advantage = r_t * estimate_advantage
    clip_advantage = torch.clip(r_t, 1. - epsilon, 1. + epsilon) * estimate_advantage
    output_clip = torch.min(r_advantage, clip_advantage)
    return - torch.mean(output_clip, dim=0)


class GaussianActor(Actor):
    def __init__(self, state_dim, action_dim, std_init=1., std_decay=0.993):
        super(GaussianActor, self).__init__()
        self.std = std_init * np.ones(action_dim, dtype=np.float32)
        self.std_decay = std_decay
        self.linear_mlp_stack = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.linear_mlp_stack(x)
        return mu

    def distribution(self, x: torch.Tensor) -> (np.ndarray, np.ndarray):
        with torch.no_grad():
            mu = self.forward(x).cpu().numpy()
        return mu, self.std

    def update_std(self):
        self.std *= self.std_decay


class MLPCritic(Critic):
    def __init__(self, state_dim):
        super(MLPCritic, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        # x = self.flatten(x)
        value = self.linear_stack(x)
        return value


class PPO_Agent(Agent):
    def __init__(self, state_dim, action_dim):
        super(PPO_Agent, self).__init__('PPO', './data/models')
        if len(self.hyperparameter.keys()) == 0:
            self.hyperparameter['actor_lr'] = 3e-4
            self.hyperparameter['actor_batch_size'] = 64
            self.hyperparameter['actor_mini_epoch_num'] = 10
            self.hyperparameter['critic_lr'] = 1e-4
            self.hyperparameter['critic_batch_size'] = 64
            self.hyperparameter['critic_mini_epoch_num'] = 15
            self.hyperparameter['discounted_rate'] = 0.99
            self.hyperparameter['actor_lr_decay'] = 0.993
            self.hyperparameter['critic_lr_decay'] = 0.993
            self.hyperparameter['epsilon'] = 0.2
            self.actor = GaussianActor(state_dim, action_dim, 1.)
            self.critic = MLPCritic(state_dim)
        self.delta = GAE()

    def reaction(self, state: np.ndarray):
        state_tensor = torch.from_numpy(state)
        with torch.no_grad():
            self.actor.to('cpu')
            mu, std = self.actor.distribution(state_tensor)
            action = np.random.normal(mu, std)
            return action, mu, std

    def update_critic(self, epoch_num: int, data: DataBuffer, device: str, log_writer: SummaryWriter):
        reward_np = data['reward']
        termination_np = data['termination']
        discounted_return = discount_cumulate(reward_np, termination_np,
                                              self.hyperparameter['discounted_rate'])
        exp_data = {
            'state': data['state'],
            'G': discounted_return
        }
        # dataset = RLDataset(exp_data, data.max_size)
        # data_loader = DataLoader(dataset, batch_size=self.hyperparameter['critic_batch_size'],
        #                          shuffle=True, drop_last=True)
        # self.critic.to(device)
        # optimizer = torch.optim.SGD(self.critic.parameters(), lr=self.hyperparameter['critic_lr'])
        # loss = torch.nn.MSELoss()
        # average_residual = 0
        # # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # for i in range(self.hyperparameter['critic_mini_epoch_num']):
        #     for batch_i, data_i in enumerate(data_loader):
        #         current_state = data_i['state'].to(device)
        #         value = data_i['G'].to(device)
        #         estimate_value = self.critic(current_state)
        #         residual = loss(estimate_value, value)
        #         optimizer.zero_grad()
        #         residual.backward()
        #         optimizer.step()
        #         average_residual += residual.item()
        batch_size = self.hyperparameter['critic_batch_size']
        mini_epoch_num = self.hyperparameter['critic_mini_epoch_num']
        exp_data['state'], exp_data['G'] = shuffle(exp_data['state'].repeat(mini_epoch_num, axis=0),
                                                   exp_data['G'].repeat(mini_epoch_num, axis=0))
        exp_data['state'] = torch.as_tensor(exp_data['state'], dtype=torch.float32).to(device)
        exp_data['G'] = torch.as_tensor(exp_data['G'], dtype=torch.float32).to(device)
        self.critic.to(device)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.critic.parameters(), lr=self.hyperparameter['critic_lr'])
        average_residual = 0
        j = 0
        if (j + 1) * batch_size <= len(exp_data['state']):
            current_state = exp_data['state'][j*batch_size:(j+1)*batch_size]
            value = exp_data['G'][j*batch_size:(j+1)*batch_size]
            estimate_value = self.critic(current_state)
            residual = loss(estimate_value, value)
            optimizer.zero_grad()
            residual.backward()
            optimizer.step()
            average_residual += residual.item()
            j += 1

        # print log
        exp_data = {
            'state': data['state'],
            'G': discounted_return
        }
        dataset = RLDataset(exp_data, data.max_size)
        if epoch_num % 200 == 0:
            average_residual /= (self.hyperparameter['critic_mini_epoch_num'] *
                                 int(len(dataset) / self.hyperparameter['critic_batch_size']))
            print_time()
            print('\t\t regression state value for advantage; epoch: ' + str(epoch_num))
            print('\t\t value loss: ' + str(average_residual))
            print('-----------------------------------------------------------------')
            log_writer.add_scalar('value loss', average_residual, epoch_num)

        value_data_array = None
        # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=40000,
        #                                              shuffle=False, num_workers=0, drop_last=False)
        with torch.no_grad():
            # for batch_i, data_i in enumerate(dataset_loader):
            current_state = torch.as_tensor(exp_data['state'], dtype=torch.float32).to(device)
            state_value = self.critic(current_state)
            if value_data_array is None:
                value_data_array = state_value.cpu().numpy()
            else:
                value_data_array = np.vstack((value_data_array, state_value.cpu().numpy()))
        data.insert('state_value', value_data_array)

    def update_actor(self, epoch_num: int, data: DataBuffer, device: str, log_writer: SummaryWriter):
        self.actor.train()
        self.actor.to(device)
        # calculation advantage
        gae_np = self.delta(data['reward'], data['state_value'], data['termination'])
        exp_data = {
            'state': data['state'],
            'action': data['action'],
            'action_likelihood': data['action_likelihood'],
            'GAE': gae_np
        }
        # dataset = RLDataset(exp_data, data.max_size)
        # dataset_loader = DataLoader(dataset, batch_size=self.hyperparameter['actor_batch_size'], shuffle=False,
        #                             num_workers=0, drop_last=True)
        # optimizer_policy = torch.optim.SGD(self.actor.parameters(), lr=self.hyperparameter['actor_lr'])
        # gaussian_normalize = torch.tensor(1. / (self.actor.std * np.sqrt(2 * np.pi)), dtype=torch.float32).to(
        #     device)
        # action_std_vers = torch.tensor(1. / self.actor.std, dtype=torch.float32).to(device)
        # for i in range(self.hyperparameter['actor_mini_epoch_num']):
        #     for batch_i, data_i in enumerate(dataset_loader):
        #         current_state = data_i['state'].to(device)
        #         action = data_i['action'].to(device)
        #         action_lh_old = data_i['action_likelihood'].to(device)
        #         mu = self.actor(current_state)
        #         action_lh_new = gaussian_normalize * torch.exp(
        #             torch.mul(-0.5, torch.pow((action - mu) * action_std_vers, 2)))
        #         advantage = data_i['GAE'].to(device)
        #         loss = loss_function(action_lh_new, action_lh_old, advantage, self.hyperparameter['epsilon'])
        #         optimizer_policy.zero_grad()
        #         loss.backward(torch.ones_like(loss))
        #         optimizer_policy.step()
        #
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        batch_size = self.hyperparameter['actor_batch_size']
        mini_epoch_num = self.hyperparameter['actor_mini_epoch_num']
        exp_data['state'], exp_data['action'], exp_data['action_likelihood'], exp_data['GAE'] = \
            shuffle(exp_data['state'].repeat(mini_epoch_num, axis=0),
                    exp_data['action'].repeat(mini_epoch_num, axis=0),
                    exp_data['action_likelihood'].repeat(mini_epoch_num, axis=0),
                    exp_data['GAE'].repeat(mini_epoch_num, axis=0))

        exp_data['state'] = torch.as_tensor(exp_data['state'], dtype=torch.float32).to(device)
        exp_data['action'] = torch.as_tensor(exp_data['action'], dtype=torch.float32).to(device)
        exp_data['action_likelihood'] = torch.as_tensor(exp_data['action_likelihood'],
                                                        dtype=torch.float32).to(device)
        exp_data['GAE'] = torch.as_tensor(exp_data['GAE'], dtype=torch.float32).to(device)
        j = 0
        optimizer_policy = torch.optim.SGD(self.actor.parameters(), lr=self.hyperparameter['actor_lr'])
        gaussian_normalize = torch.tensor(1. / (self.actor.std * np.sqrt(2 * np.pi)), dtype=torch.float32).to(
            device)
        action_std_vers = torch.tensor(1. / self.actor.std, dtype=torch.float32).to(device)
        while (j+1)*batch_size <= len(exp_data['state']):
            current_state = exp_data['state'][j*batch_size:(j+1)*batch_size]
            action = exp_data['action'][j*batch_size:(j+1)*batch_size]
            action_lh_old = exp_data['action_likelihood'][j*batch_size:(j+1)*batch_size]
            mu = self.actor(current_state)
            action_lh_new = gaussian_normalize * torch.exp(
                torch.mul(-0.5, torch.pow((action - mu) * action_std_vers, 2)))
            advantage = exp_data['GAE'][j*batch_size:(j+1)*batch_size]
            loss = loss_function(action_lh_new, action_lh_old, advantage, self.hyperparameter['epsilon'])
            optimizer_policy.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer_policy.step()
            j += 1
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))


class PPO_exp(RLExperiment):
    def __init__(self, env, state_dim, action_dim, agent,
                 buffer_size, log_path=None):
        buffer_template = {'state': state_dim, 'action': action_dim,
                           'action_likelihood': action_dim, 'reward': 0, 'termination': 0}
        super(PPO_exp, self).__init__(env, 0.99, agent, buffer_size, buffer_template, log_path)
        # self.buffer = DataBuffer(buffer_size, data_template)
        self.trajectory_size = buffer_size
        self.reward_std = None
        self.state_std = None
        self.state_mean = None
        self.env_info('./data/models/')

    def generate_trajectories(self, total_step_num: int):
        step_num = 0
        current_state = self.env.reset()
        while step_num < total_step_num:
            current_state = np.float32((current_state - self.state_mean) / self.state_std)
            action, mu, std = self.agent.reaction(current_state)
            gaussian_normalize = 1. / (std * np.sqrt(2 * np.pi))
            action_likelihood = gaussian_normalize * np.exp(-0.5 * ((mu - action) / std) ** 2)
            new_state, reward, is_done, _ = self.env.step(action)
            self.buffer.push([current_state, action, action_likelihood, reward, is_done])
            step_num += 1
            if is_done:
                current_state = self.env.reset()
            else:
                current_state = new_state
        self.buffer['reward'] = self.buffer['reward'] / self.reward_std

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
        env = self.env
        total_reward = 0.0
        total_steps = 0
        self.agent.actor.eval()
        self.agent.actor.to(device)
        for i in range(test_round_num):
            obs = env.reset()
            while True:
                obs = np.float32((obs - self.state_mean) / self.state_std)
                obs_tensor = torch.from_numpy(obs).to(device)
                action = self.agent.actor(obs_tensor).detach().numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                total_steps += 1
                if done:
                    break
        print("Episode done in %d steps, total reward %.2f" % (
            total_steps / test_round_num, total_reward / test_round_num))
        env.close()
        self.exp_log_writer.add_scalar('reward', total_reward / test_round_num, round_num)
        self.exp_log_writer.add_scalar('step', total_steps / test_round_num, round_num)
        return total_reward / test_round_num

    def play(self, round_num=1000000):
        start_round_num = agent.start_epoch
        print_time()
        for round_i in range(start_round_num, round_num):

            self.generate_trajectories(total_step_num=self.trajectory_size)
            agent.update_critic(round_i, self.buffer, 'cpu', self.exp_log_writer)
            # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            agent.update_actor(round_i, self.buffer, 'cpu', self.exp_log_writer)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

            if round_i % 1000 == 0:
                self.test(round_i, 100, 'cpu')
                self.save(round_i)
                self.agent.actor.update_std()
                self.agent.hyperparameter['actor_lr'] = max(self.agent.hyperparameter['actor_lr'] *
                                                            self.agent.hyperparameter['actor_lr_decay'], 1e-6)
                self.agent.hyperparameter['critic_lr'] = max(self.agent.hyperparameter['critic_lr'] *
                                                             self.agent.hyperparameter['critic_lr_decay'], 1e-6)
                self.exp_log_writer.add_scalar('value lr', self.agent.hyperparameter['critic_lr'], round_i)
                self.exp_log_writer.add_scalar('policy lr', self.agent.hyperparameter['actor_lr'], round_i)
                self.exp_log_writer.add_scalar('action std scaled', self.agent.actor.std[0], round_i)


# Hyperparameter          Value
# Horizon (T)             2048
# Adam step size          3e−4
# Num. epochs               10
# Minibatch size             64
# Discount ( γ )           0.99
# generalized_advantage_estimation parameter (λ)         0.95


if __name__ == '__main__':
    env_ = gym.make('Swimmer-v3')
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    agent = PPO_Agent(state_dim_, action_dim_)
    experiment = PPO_exp(env_, state_dim_, action_dim_, agent, 2048, './data/log/')
    experiment.play(round_num=1000000)
    env_.close()
