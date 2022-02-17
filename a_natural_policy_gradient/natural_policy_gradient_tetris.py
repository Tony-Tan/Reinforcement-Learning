# import simplified_tetris as tetris
import tetris
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
import copy

writer = SummaryWriter('./data/log/')


class PolicyFunction:
    def __init__(self, input_size, weights_path=None):
        self.action_space_size = 4
        dimension = [self.action_space_size, input_size + 4]
        if weights_path is None:
            self.weights = torch.randn(dimension, requires_grad=True)
        else:
            self.weights = torch.load(weights_path)

    def select_action(self, prob):
        return np.random.choice(range(self.action_space_size), 1, p=prob)[0]


def vililla_policy_gradient(alpha=0.001,  gamma=0.9, natural_policy_gradient_=False):
    tetris_width = 6
    tetris_height = 8
    env = tetris.Tetris(tetris_width, tetris_height)
    # policy_f = PolicyFunction((tetris_width*tetris_height))
    policy_f = PolicyFunction((tetris_width))
    reward_list = []
    total_reward = 0
    for epoch_i in range(1, 100000000):
        state, reward, is_done, _ = env.reset()
        trace = torch.zeros(policy_f.weights.shape, requires_grad=False)
        delta_eta = torch.zeros(policy_f.weights.shape, requires_grad=False)

        step_times = 0
        fisher_matrix = 0
        while not is_done:
            # state_bg = state[0].reshape(1, -1).astype(np.float32) - 0.5
            heights_arr = (state[0] != 0).argmax(axis=0)
            state_t = np.array([state[1], state[2] / 360., state[3][0] / tetris_width,
                                state[3][1] / tetris_height]).astype(np.float32)
            state_ = np.append(heights_arr, state_t)
            # state_ = np.append(state_bg, state_t)
            x_tensor = torch.from_numpy(state_).requires_grad_()
            # compute the policy distribution
            prob = F.softmax(torch.sum(torch.mul(x_tensor, policy_f.weights), dim=1), dim=0).requires_grad_()

            prob_to_selection = prob.clone().detach()
            action = policy_f.select_action(prob_to_selection.numpy())

            next_state, reward, is_done, _ = env.step_autofill(action)

            # frame = env.draw()
            # cv2.imshow('play', frame)
            # cv2.waitKey(1)
            gradient = torch.autograd.grad(outputs=prob[action], inputs=policy_f.weights, retain_graph=True)[0]
            if natural_policy_gradient_:
                prob_action = prob[action]
                gradient_num = gradient.clone().detach().numpy().reshape((1, -1))/prob_action.detach().numpy()
                fisher_matrix += gradient_num.transpose().dot(gradient_num)

            trace = trace + gradient/prob[action]
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
                fisher_matrix_inv /= np.max(fisher_matrix_inv)
                policy_weights_delta = fisher_matrix_inv.dot(
                            delta_eta.detach().numpy().reshape(-1, 1)).reshape(policy_f.action_space_size, -1)
                policy_f.weights = policy_f.weights + alpha * torch.from_numpy(policy_weights_delta)
            else:
                policy_f.weights = policy_f.weights + alpha * delta_eta
            policy_f.weights.requires_grad_()
            # print(trace)
        if epoch_i % 10000 == 0:
            print('--------------epoch: %d ------------------\n' % epoch_i)
            print('total reward: %d' % total_reward)
            reward_list.append(total_reward)
            writer.add_scalar('reward', total_reward/10000., epoch_i)
            total_reward = 0
        if epoch_i % 10000 == 0:
            torch.save(policy_f.weights, './data/weights.pt')
            # print(policy_f.weights)


if __name__ == '__main__':
    vililla_policy_gradient(natural_policy_gradient_=False)
