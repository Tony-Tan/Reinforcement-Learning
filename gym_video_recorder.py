"""
record training process of agent
load all time models from the fold and calculating the average reward and step
record one episode by video recorder of opencv
"""
import gym, mujoco_py
from sac.sac import *
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

device = 'cpu'
env_name = 'Walker2d-v3'

if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('./sac/data/exp/' + env_name + '.mp4', fourcc, 20.0, (2048, 2048))
    test_times = 100
    env_ = gym.make(env_name)
    agent_ = SAC_Agent(env_.observation_space, env_.action_space, [256, 256],
                       [256, 256], 'sac/data/models')
    agent_.load()
    agent_.actor.to(device)
    experiment = SAC_exp(env_, agent_, 1, 0.99, False, 'sac/data/log', env_data_path='sac/data/models')
    total_reward = 0
    total_steps = 0
    for i in range(test_times):
        obs = env_.reset()
        while True:
            if experiment.normalize:
                x_tensor = torch.tensor(((obs - experiment.state_mean) / experiment.state_std), dtype=torch.float32)
            else:
                x_tensor = torch.as_tensor(obs,dtype=torch.float32)
            agent_.actor.eval()
            # mu = agent.actor(x_tensor)
            action, _ = agent_.actor.act(x_tensor,stochastically=False)
            action = action.detach().cpu().numpy()
            obs, reward, done, _ = env_.step(action)
            # print(obs)
            total_reward += reward
            total_steps += 1
            # env.render()
            if done:
                break

    obs = env_.reset()
    done = False
    while not done:
        if experiment.normalize:
            x_tensor = torch.tensor(((obs - experiment.state_mean) / experiment.state_std), dtype=torch.float32)
        else:
            x_tensor = torch.as_tensor(obs, dtype=torch.float32)
        mu = agent_.actor(x_tensor)[0]
        action = mu.detach().cpu().numpy()
        obs, reward, done, _ = env_.step(action)
        screen = env_.render(mode='rgb_array')
        screen = cv2.resize(screen, [2048, 2048])
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(screen, 'Epoch: %d' % agent_.start_epoch, (30, 140), font, 2.8,
                    (0, 0, 200), 8, cv2.LINE_AA)
        cv2.putText(screen, 'Average Reward: %0.3f' % (total_reward / test_times), (30, 260), font, 2.8,
                    (0, 0, 200), 8, cv2.LINE_AA)
        cv2.putText(screen, 'Average Step: %0.2f' % (total_steps / test_times), (30, 380), font, 2.8,
                    (0, 0, 200), 8, cv2.LINE_AA)
        out.write(screen)
        if done:
            break
    env_.close()
    out.release()
