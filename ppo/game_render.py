import gym, mujoco_py
from ppo import PolicyNN, ValueNN
import ppo
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
device = 'cpu'
env_name = 'InvertedDoublePendulum-v2'

if __name__ == "__main__":
    agent = ppo.PPOAgent(env_name)
    # ppo.STD_DEVIATION = 0.5
    env = gym.make(env_name)
    # env = gym.wrappers.Monitor(env, "recording")

    total_reward = 0.0
    total_steps = 0
    action_list = []
    for i in range(1000):
        obs = env.reset()
        while True:
            x_tensor = torch.tensor(((obs-agent.state_mean)/agent.state_std), dtype=torch.float32).to(device)
            agent.policy_module.eval()

            mu = agent.policy(x_tensor)
            action = mu.clone().detach().cpu().numpy()
            # action = agent.action_selection(x_tensor)[0][0]
            action_list.append(action)
            obs, reward, done, _ = env.step(action)
            # print(obs)
            total_reward += reward
            total_steps += 1
            # env.render()
            if done:
                # print('new episode:')
                break
    plt.hist(np.array(action_list), bins=20, color='red', histtype='stepfilled', alpha=0.75, density=True)
    plt.title('action distribution')
    plt.show()
    print("Episode done in %d steps, total reward %.2f" % (
        total_steps/1000, total_reward/1000))
    env.close()
