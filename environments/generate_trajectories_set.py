# generate trajectories set
# actions are coded by non-negative int number: 0,1.....len(policy)
import numpy as np
import sys
import torch


def select_action(policy):
    action = np.random.choice(range(len(policy)), 1, p=policy)[0]
    return action


def generate_trajectory_set(env, set_size, policy_fn, feature_fn, max_trajectory_length=sys.maxsize, device='cpu'):
    total_reward = 0
    trajectory_collection = []
    with torch.no_grad():
        for set_i in range(set_size):
            trajectory_i = []
            current_state, reward, is_done, _ = env.reset()
            current_state_feature = torch.tensor(feature_fn(current_state), dtype=torch.float32,
                                                 requires_grad=False).to(device)

            while not is_done:
                if len(trajectory_i) > max_trajectory_length:
                    break
                # generate experience
                try:
                    action = select_action(policy_fn(current_state_feature))
                except ValueError:
                    return None,None
                next_state, reward, is_done, _ = env.step(action)
                next_state_feature = torch.tensor(feature_fn(next_state), dtype=torch.float32,
                                                  requires_grad=False).to(device)
                trajectory_i.append([current_state_feature, action, reward])
                current_state_feature = next_state_feature

                total_reward += reward
            trajectory_collection.append(trajectory_i)
    return trajectory_collection, total_reward
