import natural_policy_gradient_tetris as pt
import torch
import tetris
import cv2
import numpy as np
import torch.nn.functional as F


tetris_width = 6
tetris_height = 10
env = tetris.Tetris(tetris_width, tetris_height)
policy_f = pt.PolicyFunction((tetris_width, tetris_height),weights_path='./data/weights.pt')
total_reward = 0
state, reward, is_done, _ = env.reset()
while not is_done:
    state_bg = state[0].reshape(1, -1).astype(np.float32) - 0.5
    # heights_arr = (state[0] != 0).argmax(axis=0)
    state_t = np.array([state[1], state[2] / 360., state[3][0] / tetris_width,
                        state[3][1] / tetris_height]).astype(np.float32)
    # state_ = np.append(heights_arr, state_t)
    state_ = np.append(state_bg, state_t)
    x_tensor = torch.from_numpy(state_).requires_grad_(False)
    # compute the policy distribution
    prob = F.softmax(torch.sum(torch.mul(x_tensor, policy_f.weights), dim=1), dim=0).requires_grad_()
    prob_to_selection = prob.clone().detach()
    action = policy_f.select_action(prob_to_selection.numpy())
    next_state, reward, is_done, _ = env.step_autofill(action)

    frame = env.draw()
    cv2.imshow('play', frame)
    cv2.waitKey(100)

    total_reward += reward
    state = next_state
    print(total_reward)
