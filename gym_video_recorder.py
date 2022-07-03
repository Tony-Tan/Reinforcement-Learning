"""
record training process of agent
load all time models from the fold and calculating the average reward and step
record one episode by video recorder of opencv
"""
import gym, mujoco_py
from DDPG.ddpg import *
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

device = 'cpu'
env_name = 'Walker2d-v3'

if __name__ == "__main__":
    test_times = 100
    model_list = os.listdir('DDPG/data/models/')
    model_list.sort()
    epoch = 2000
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('DDPG/data/exp/' + env_name + '.mp4', fourcc, 20.0, (2048, 2048))
    env_ = gym.make(env_name)
    state_dim_ = env_.observation_space.shape[-1]
    action_dim_ = env_.action_space.shape[-1]
    agent = DDPG_Agent(state_dim_, action_dim_, [64, 64], [64, 64],  './DDPG/data/models')
    experiment = DDPG_exp(env_, state_dim_, action_dim_, agent, 1, True,
                         log_path='DDPG/data/log', env_data_path='DDPG/data')
    for name in model_list:
        if '.pt' in name and '_actor' in name:
            epoch += 2000
            if epoch % 20000 != 0:
                continue
            print('loading model: ' + name)
            model_name = name.split('_')[0]
            agent.load(model_name=model_name)
            total_reward = 0.0
            total_steps = 0
            action_list = []
            for i in range(test_times):
                obs = env_.reset()
                while True:
                    if experiment.normalize:
                        x_tensor = torch.tensor(((obs - experiment.state_mean) / experiment.state_std), dtype=torch.float32)
                    else:
                        x_tensor = torch.as_tensor(obs,dtype=torch.float32)
                    agent.actor.eval()
                    # mu = agent.actor(x_tensor)
                    mu = agent.actor(x_tensor)[0]
                    action = mu.detach().cpu().numpy()
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
                mu = agent.actor(x_tensor)[0]
                action = mu.detach().cpu().numpy()
                obs, reward, done, _ = env_.step(action)
                screen = env_.render(mode='rgb_array')
                screen = cv2.resize(screen, [2048, 2048])
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(screen, 'Epoch: %d' % (epoch), (30, 140), font, 2.8,
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
