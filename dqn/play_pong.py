import gym
import alg.Network as Network
import cv2
import torch.utils.data as utils_data
import torch
import numpy as np
import os
import random


class AgentDemo:
    def __init__(self, env, k_frames=4, phi_temp_size=4, model_path='./model/', video_output_path='./'):
        self.env = env
        self.output_video_path = video_output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 840))
        self._action_n = self.env.action_space.n
        self.state_value_function = Network.Net(4, self._action_n)
        self.k_frames = k_frames
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_value_function.to(self._device)
        self.down_sample_size = 84
        self.model_path = model_path
        self.phi_temp = []
        self.phi_temp_size = phi_temp_size
        if os.path.exists(self.model_path + 'trained/pong-v0.pth'):
            self.state_value_function.load_state_dict(torch.load(self.model_path + 'trained/pong-v0.pth'))

    def convert_down_sample(self, state):
        image = np.array(state)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (84, 100))
        return gray_img[100 - 84:100, 0:84] / 255. - 0.5

    def select_action(self, state_phi, epsilon):
        state_phi_tensor = torch.from_numpy(state_phi).unsqueeze(0).to(self._device)
        state_action_values = self.state_value_function(state_phi_tensor).cpu().detach().numpy()
        value_of_action_list = state_action_values[0]
        optimal_action = np.random.choice(
            np.flatnonzero(value_of_action_list == value_of_action_list.max()))
        if random.randint(0, 1000) < epsilon * 1000.:
            return random.randint(0, self._action_n - 1)
        else:
            return optimal_action

    def phi(self):
        if len(self.phi_temp) > self.phi_temp_size:
            self.phi_temp = self.phi_temp[len(self.phi_temp) - self.phi_temp_size:len(self.phi_temp)]
        return (np.array(self.phi_temp)).astype(np.float32)

    def skip_k_frame(self, action):
        new_state = 0
        reward = 0
        is_done = 0
        others = 0
        for i in range(self.k_frames):
            new_state, r, is_done, others = self.env.step(action)
            # display and record each state
            cv2.imshow('state', new_state)
            image_into_video = cv2.resize(new_state, (320, 420))
            self.video_writer.write(image_into_video)
            cv2.waitKey(10)
            reward += r
            if is_done:
                break
        return new_state, reward, is_done, others

    def play(self, episodes_num):
        for episode_i in range(1, episodes_num):
            action = np.random.choice(self._action_n, 1)[0]
            state = self.env.reset()
            state = self.convert_down_sample(np.array(state))
            self.phi_temp.append(state)
            for i in range(1, 4):
                new_state, reward, is_done, _ = self.skip_k_frame(action)
                new_state = self.convert_down_sample(np.array(new_state))
                self.phi_temp.append(new_state)
            # create phi
            state_phi = self.phi()
            # select action according the first phi
            action = self.select_action(state_phi, epsilon=0.005)
            new_state, reward, is_done, _ = self.skip_k_frame(action)
            # down sample and add the new state into the list to create phi
            new_state = self.convert_down_sample(np.array(new_state))
            self.phi_temp.append(new_state)
            new_state_phi = self.phi()
            while not is_done:
                state_phi = new_state_phi
                action = self.select_action(state_phi, epsilon=0.005)
                new_state, reward, is_done, _ = self.skip_k_frame(action)

                new_state = self.convert_down_sample(np.array(new_state))
                self.phi_temp.append(new_state)
                new_state_phi = self.phi()
        return


if __name__ == '__main__':
    env = gym.make('Pong-v0')
    agent = AgentDemo(env)
    agent.play(10)
