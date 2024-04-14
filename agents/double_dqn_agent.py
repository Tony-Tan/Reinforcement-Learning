import random

from agents.dqn_agent import *


class DoubleDQNValueFunction(DQNValueFunction):
    def __init__(self, input_channel: int, action_dim: int, learning_rate: float,
                 gamma: float, step_c: int, model_saving_period: int, device: torch.device, logger: Logger):
        super(DoubleDQNValueFunction, self).__init__(input_channel, action_dim, learning_rate,
                                                     gamma, step_c, model_saving_period, device, logger)
        self.target_value_nn.train()
        self.optimizer_nn = torch.optim.Adam(self.value_nn.parameters(), lr=learning_rate)
        self.optimizer_tnn = torch.optim.Adam(self.target_value_nn.parameters(), lr=learning_rate)

    def update(self, samples: list):
        obs_tensor = image_normalization(samples[0])
        action_tensor = samples[1]
        reward_tensor = samples[2]
        termination_tensor = samples[4]
        truncated_tensor = samples[5]
        next_obs_tensor = image_normalization(samples[3])
        target_is_target = True
        # select the target nn and value nn
        if random.random() < 0.5:
            target_is_target = False

        with torch.no_grad():
            outputs_tnn = self.target_value_nn(next_obs_tensor)
            outputs_nn = self.value_nn(next_obs_tensor)
        if target_is_target:
            _, greedy_actions = torch.max(outputs_tnn, dim=1, keepdim=True)
            max_next_state_value = outputs_nn.gather(1, greedy_actions)
        else:
            _, greedy_actions = torch.max(outputs_nn, dim=1, keepdim=True)
            max_next_state_value = outputs_tnn.gather(1, greedy_actions)
        reward_tensor.resize_as_(max_next_state_value)
        # calculate q value
        truncated_tensor.resize_as_(max_next_state_value)
        termination_tensor.resize_as_(max_next_state_value)
        q_value = reward_tensor + self.gamma * max_next_state_value * (1 - truncated_tensor) * (1 - termination_tensor)
        action_tensor.resize_as_(reward_tensor)
        q_value.resize_as_(reward_tensor)
        actions = action_tensor.long()
        if target_is_target:
            self.optimizer_nn.zero_grad()
        else:
            self.optimizer_tnn.zero_grad()
        if target_is_target:
            outputs = self.value_nn(obs_tensor)
        else:
            outputs = self.target_value_nn(obs_tensor)
        obs_action_value = outputs.gather(1, actions)
        # loss = F.mse_loss(q_value, obs_action_value)
        # Clip the difference between obs_action_value and q_value to the range of -1 to 1
        diff = obs_action_value - q_value
        diff_clipped = torch.clip(diff, -1, 1)

        # Use the clipped difference for the loss calculation
        loss = F.mse_loss(diff_clipped, torch.zeros_like(diff_clipped))
        loss.backward()
        if target_is_target:
            self.optimizer_nn.step()
        else:
            self.optimizer_tnn.step()
        self.update_step += 1
        if self.update_step % self.step_c == 0:
            self.synchronize_value_nn()
            self.logger.tb_scalar('loss', loss.item(), self.update_step)
            self.logger.tb_scalar('q', torch.mean(q_value), self.update_step)

    def value(self, phi_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            phi_tensor = phi_tensor.to(self.device)
            if phi_tensor.dim() == 3:
                obs_input = phi_tensor.unsqueeze(0)
            else:
                obs_input = phi_tensor
            if random.random() < 0.5:
                state_action_values = self.target_value_nn(obs_input).cpu().detach().numpy()
            else:
                state_action_values = self.value_nn(obs_input).cpu().detach().numpy()
            return state_action_values


class DoubleDQNAgent(DQNAgent):
    def __init__(self, input_frame_width: int, input_frame_height: int, action_space,
                 mini_batch_size: int, replay_buffer_size: int, min_update_sample_size: int,
                 learning_rate: float, step_c: int, model_saving_period: int,
                 gamma: float, training_episodes: int, phi_channel: int, epsilon_max: float, epsilon_min: float,
                 exploration_steps: int, device: torch.device, logger: Logger):
        super(DoubleDQNAgent, self).__init__(input_frame_width, input_frame_height, action_space,
                                             mini_batch_size, replay_buffer_size, min_update_sample_size,
                                             learning_rate, step_c, model_saving_period,
                                             gamma, training_episodes, phi_channel, epsilon_max, epsilon_min,
                                             exploration_steps, device, logger)
        self.value_function = DoubleDQNValueFunction(phi_channel, action_space.n, learning_rate,
                                                     gamma, step_c, model_saving_period, device, logger)

