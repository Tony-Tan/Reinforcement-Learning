from environments import simplified_tetris as tetris
# import tetris
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter('../a_natural_policy_gradient/data/log/')


class PolicyFunction:
    def __init__(self, env, input_size, weights_path=None):
        self.action_space_size = len(env.action_space)
        dimension = [self.action_space_size, input_size + 5]
        if weights_path is None:
            self.weights = torch.zeros(dimension, requires_grad=True,dtype=torch.float32)
        else:
            print('load weights from: ' + weights_path)
            self.weights = torch.load(weights_path)
            print(self.weights)
        np.random.seed(int(time.time()))

    def select_action(self, prob):
        return np.random.choice(range(self.action_space_size), 1, p=prob)[0]

    def save_weights(self, path='./data/'):
        time_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        torch.save(self.weights, './data/' + time_name + '.pt')
        # print(policy_f.weights)


def vinilla_policy_gradient(alpha=0.1, gamma=0.9, natural_policy_gradient_=False, weights_path=None):
    tetris_width = 6
    tetris_height = 5
    env = tetris.Tetris(tetris_width, tetris_height)
    policy_f = PolicyFunction(env, tetris_width * tetris_height, weights_path)
    # policy_f = PolicyFunction(env, tetris_width * 2 - 1, weights_path)
    reward_list = []
    total_reward = 0
    action_num = len(env.action_space)
    policy_scale = 0.9999
    for epoch_i in range(1, 1000000):

        state, reward, is_done, _ = env.reset()
        trace = torch.zeros(policy_f.weights.shape, requires_grad=False)
        delta_eta = torch.zeros(policy_f.weights.shape, requires_grad=False)

        step_times = 1
        fisher_matrix = 0

        while not is_done:
            # if step_times > 1000:
            #     print('too good to train')
            #     policy_f.save_weights()
            #     return
            state_bg = (state[0].reshape(1, -1).astype(np.float32) - 0.5) * 2
            # heights_arr = (state[0] != 0).argmax(axis=0) / tetris_height
            # diff_height = np.array(
            #     [heights_arr[i] - heights_arr[i + 1] for i in range(len(heights_arr) - 1)]) / tetris_height
            # the last number is for bias
            state_t = np.array([state[1], state[2] / 360., state[3][0] / tetris_width,
                                state[3][1] / tetris_height, 1.0]).astype(np.float32)
            # state_ = np.append(heights_arr, diff_height)
            # state_ = np.append(state_, state_t)
            state_ = np.append(state_bg, state_t)
            x_tensor = torch.from_numpy(state_).requires_grad_()
            # compute the policy distribution
            prob = F.softmax(torch.sum(torch.mul(x_tensor, policy_f.weights), dim=1), dim=0).requires_grad_()
            prob = torch.add(torch.mul(prob, policy_scale), (1 - policy_scale) / action_num)
            prob_to_selection = prob.clone().detach().numpy()

            action = policy_f.select_action(prob_to_selection)

            next_state, reward, is_done, _ = env.step_autofill(action)

            # frame = env.draw()
            # cv2.imshow('play', frame)
            # cv2.waitKey(10)
            gradient = None
            if natural_policy_gradient_:
                prob_num = prob.clone().detach().numpy()
                for action_i in range(policy_f.action_space_size):
                    if action == action_i:
                        gradient = torch.autograd.grad(outputs=prob[action], inputs=policy_f.weights, retain_graph=True)[0]
                    else:
                        gradient_temp = torch.autograd.grad(outputs=prob[action], inputs=policy_f.weights, retain_graph=True)[0]
                        gradient_num = gradient_temp.detach().numpy().reshape((1, -1)) / prob_num[action_i]
                        fisher_matrix += (gradient_num.transpose().dot(gradient_num)*prob_num[action_i])
            else:
                gradient = torch.autograd.grad(outputs=prob[action], inputs=policy_f.weights, retain_graph=True)[0]
            trace = trace + gradient / prob[action]
            delta_eta = delta_eta + trace * reward
            total_reward += reward

            state = next_state
            step_times += 1

        with torch.no_grad():
            delta_eta /= step_times
            if natural_policy_gradient_:
                fisher_matrix /= step_times
                # if np.max(fisher_matrix) != 0:
                #     fisher_matrix /= np.max(fisher_matrix)
                fisher_matrix += np.eye(np.shape(fisher_matrix)[0]) * 0.001
                fisher_matrix_inv = np.linalg.inv(fisher_matrix)
                # fisher_matrix_inv /= np.max(fisher_matrix_inv)
                policy_weights_delta = fisher_matrix_inv.dot(
                    delta_eta.detach().numpy().reshape(-1, 1)).reshape(policy_f.action_space_size, -1)
                policy_f.weights = policy_f.weights + torch.tensor(policy_weights_delta, dtype=torch.float32)
            else:
                policy_f.weights = policy_f.weights + alpha * delta_eta
            policy_f.weights.requires_grad_()
            # print(trace)
        if epoch_i % 1000 == 0:
            print('--------------epoch: %d ------------------\n' % epoch_i)
            print('total reward: %d' % total_reward)
            print('policy weights:')
            print(policy_f.weights)
            reward_list.append(total_reward)
            writer.add_scalar('reward', total_reward / 1000., epoch_i)
            # writer.add_scalar('policy-epsilon', policy_scale, epoch_i)
            # baseline /= 10000
            total_reward = 0
        # if epoch_i%100000 == 0:
        #     policy_scale *= 1.01
        # if epoch_i % 1000 == 0:
        #     policy_f.save_weights()


if __name__ == '__main__':
    # weights_path_ = './data/2022-02-25-14-18-30.pt'
    weights_path_ = None
    vinilla_policy_gradient(natural_policy_gradient_=True, weights_path=weights_path_)
