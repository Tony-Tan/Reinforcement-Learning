import gym
import DQN.Network as Network
import cv2
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import time
import os
import copy
import torch.nn.functional as F
from DQN.dqn import AgentDQN


class AgentDoublePP(AgentDQN):
    def __init__(self, environment, mini_batch_size=32, episodes_num=100000,
                 k_frames=4, input_frame_size=84, memory_length=2e4, phi_temp_size=4, learning_rate=1e-4,
                 alpha=0.4, beta=0.7, model_path='./model/', log_path='./log/', steps_c=50):
        super(AgentDoublePP, self).__init__(environment, mini_batch_size, episodes_num,
                                            k_frames, input_frame_size, memory_length, phi_temp_size,
                                            model_path, log_path,learning_rate, steps_c, algorithm_version='2015')
        self._alpha = alpha
        self._beta = beta
        self._weights = 0
        self.state_action_value_function_temp = copy.deepcopy(self.state_action_value_function)
        self.probability_array = deque(maxlen=int(memory_length))
        self.weights_array = deque(maxlen=int(memory_length))

    def train_network(self, memory_index, probability_array, gamma=0.99):
        """
        training the network of state value function
        :param memory_index: memory index of memory of states
        :param probability_array: memory index of memory of states
        :param gamma: float number, decay coefficient
        :return: nothing
        """
        # update weight
        prob_array = np.array([probability_array[idx] for idx in memory_index])
        weight = (self._mini_batch_size * prob_array) ** (-self._beta)
        weight /= np.max(weight)
        weight = np.sqrt(weight)

        # build label:
        next_state_data = np.array([self._memory[idx][3] for idx in memory_index])
        reward_array = np.array([self._memory[idx][2] for idx in memory_index])
        action_array = np.array([[int(self._memory[idx][1])] for idx in memory_index])
        state_data = np.array([self._memory[idx][0]*weight[i] for idx, i in zip(memory_index,range(self._mini_batch_size))])
        next_state_max_value = []
        # calculate delta
        with torch.no_grad():
            inputs = torch.from_numpy(next_state_data).to(self._device)
            outputs = self.target_state_action_value_function(inputs)
            _, predictions = torch.max(outputs, 1)
            outputs = outputs.cpu().numpy()
            predictions = predictions.cpu().numpy()
            for p_i in range(len(predictions)):
                next_state_max_value.append(outputs[p_i][predictions[p_i]])
        for idx_i in range(len(memory_index)):
            if self._memory[memory_index[idx_i]][4]:
                next_state_max_value[idx_i] = 0
        reward_array = reward_array + gamma * np.array(next_state_max_value)
        reward_array = reward_array.transpose().astype(np.float32)
        action_array = torch.Tensor(action_array).long()
        # train the model
        reward_array *= weight
        inputs = torch.from_numpy(state_data).to(self._device)
        labels = torch.from_numpy(reward_array).to(self._device).view(-1, 1)
        actions = action_array.to(self._device)
        # zero the parameter gradients
        self._optimizer.zero_grad()
        outputs = self.state_action_value_function(inputs).gather(1, actions)
        # update probability
        delta_array = np.abs((outputs - labels).cpu().detach().numpy())
        for idx, i in zip(memory_index, range(self._mini_batch_size)):
            self.probability_array[idx] = float(delta_array[i])
        # optimize
        loss = F.mse_loss(outputs, labels)
        # Minimize the loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        total_loss = loss.item()
        # record
        self._writer.add_scalar('train/loss', total_loss)

    def learning_an_episode(self, epsilon):
        frame_num = 0
        total_reward = 0
        state = self._env.reset()
        self.pre_process_and_add_state_into_phi_temp(state)

        is_done = False
        for i in range(1, self._phi_temp_size):
            action = np.random.choice(self._action_n, 1)[0]
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            self.pre_process_and_add_state_into_phi_temp(new_state)
            if is_done:
                return is_done
        state_phi = self.phi()
        while not is_done:
            action = self.select_action(state_phi, epsilon)
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            frame_num += 1
            total_reward += reward
            self.pre_process_and_add_state_into_phi_temp(new_state)
            new_state_phi = self.phi()
            self._memory.append([state_phi, action, reward, new_state_phi, is_done])
            if len(self.probability_array) == 0:
                self.probability_array.append(1.0)
            else:
                prob_array = np.array(self.probability_array)
                self.probability_array.append(np.max(prob_array))
            if len(self._memory) > self._mini_batch_size:
                prob_array = np.array(self.probability_array)
                prob_array = prob_array**self._alpha
                prob_array /= np.sum(prob_array)
                memory_index = range(0, len(self._memory))
                selected_memory_index = np.random.choice(memory_index, self._mini_batch_size, p=prob_array)
                self.train_network(memory_index=selected_memory_index, probability_array=prob_array)
            state_phi = new_state_phi
        return frame_num, total_reward

    def learning(self, epsilon_max=1.0, epsilon_min=0.1, epsilon_decay=0.9995):
        """
        :param epsilon_max: float number, epsilon start number, 1.0 for most time
        :param epsilon_min: float number, epsilon end number, 0.1 in the paper
        :param epsilon_decay: float number, decay coefficient of epsilon
        :return: nothing
        """
        frame_num = self._phi_temp_size
        epsilon = epsilon_max
        for episode_i in range(1, self._episodes_num):
            # set a dynamic epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # switch tow sets parameters of network randomly
            if random.randint(0, 1):
                self.state_action_value_function_temp.load_state_dict(
                    self.state_action_value_function.state_dict())
                self.state_action_value_function.load_state_dict(
                    self.target_state_action_value_function.state_dict())
                self.target_state_action_value_function.load_state_dict(
                    self.state_action_value_function_temp.state_dict())

            frame_num_i, reward_i = self.learning_an_episode(epsilon)
            frame_num += frame_num_i
            self.record_reward(frame_num, reward_i, epsilon, episode_i)
            if episode_i % self._steps_c == 0:
                print('------------------------ updating target state action value function -----------------------')
                self.target_state_action_value_function.load_state_dict(self.state_action_value_function.state_dict())
            if episode_i % 500 == 0:
                self.save_model()


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    agent = AgentDoublePP(env, steps_c=100)
    agent.learning()
