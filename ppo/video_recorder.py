import gym, mujoco_py
from ppo import PolicyNN, ValueNN
import ppo
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
device = 'cpu'
env_name = 'Swimmer-v2'

if __name__ == "__main__":
    test_times = 100
    model_list = os.listdir('./data/models/')
    model_list.sort()

    epoch = 1000
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('./data/exp/' + env_name + '.mp4', fourcc, 20.0, (2048, 2048))
    for name in model_list:
        if '.pt' in name and '_policy' in name:
            model_name = name.split('_')[0]
            agent = ppo.PPOAgent(env_name)
            agent.load_model(model_name, './data/models/')
            env = gym.make(env_name)
            total_reward = 0.0
            total_steps = 0
            action_list = []
            for i in range(test_times):
                obs = env.reset()
                while True:
                    x_tensor = torch.tensor(((obs-agent.state_mean)/agent.state_std), dtype=torch.float32).to(device)
                    agent.policy_module.eval()

                    mu = agent.policy(x_tensor)
                    action = mu.clone().detach().cpu().numpy()
                    # action = agent.action_selection(x_tensor)[0][0]
                    action_list.append(action[0])
                    obs, reward, done, _ = env.step(action)
                    # print(obs)
                    total_reward += reward
                    total_steps += 1
                    # env.render()
                    if done:
                        # print('new episode:')
                        break
            fig = plt.figure(figsize=(9.6, 7.2))
            plt.title('Action Distribution')
            plt.hist(np.array(action_list), bins=100, color='red', histtype='stepfilled', alpha=1., density=True)
            # plt.title('action distribution')
            # plt.savefig('./data/exp/'+model_name+'png')
            print("Episode done in %d steps, total reward %.2f" % (
                total_steps/test_times, total_reward/test_times))
            fig.canvas.draw()
            img_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img_hist = img_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img_hist = cv2.cvtColor(img_hist, cv2.COLOR_RGB2BGR)
            # cv2.imshow('img_hist', img_hist)
            # cv2.waitKey(0)

            # env = gym.wrappers.RecordVideo(env, './data/exp/', name_prefix=model_name)

            # video recording:

            obs = env.reset()
            done = False
            while not done:
                x_tensor = torch.tensor(((obs - agent.state_mean) / agent.state_std), dtype=torch.float32).to(device)
                agent.policy_module.eval()
                mu = agent.policy(x_tensor)
                action = mu.clone().detach().cpu().numpy()
                # action = agent.action_selection(x_tensor)[0][0]
                action_list.append(action)
                obs, reward, done, _ = env.step(action)
                # print(obs)
                screen = env.render(mode='rgb_array')
                screen = cv2.resize(screen, [2048, 2048])
                screen = cv2.cvtColor(screen,  cv2.COLOR_RGB2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(screen, 'Epoch: %d' % (epoch), (10, 100), font, 2,
                            (0, 0, 200), 3,
                            cv2.LINE_AA)
                cv2.putText(screen, 'Average Reward: %0.3f'%(total_reward/test_times), (10, 180), font, 2, (0, 0, 200),
                            3, cv2.LINE_AA)
                cv2.putText(screen, 'Average Step: %0.2f'%(total_steps/test_times), (10, 260), font, 2, (0, 0, 200),
                            3, cv2.LINE_AA)
                screen[1328:2048, 1088:2048] = img_hist
                cv2.imshow('screen', screen)
                cv2.waitKey(10)
                out.write(screen)
                if done:
                    # print('new episode:')
                    break
            epoch += 1000

            env.close()
    out.release()