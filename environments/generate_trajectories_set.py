# generate trajectories set
# actions are coded by non-negative int number: 0,1.....len(policy)
import numpy as np
import sys
import torch


def select_action(policy):
    action = np.random.choice(range(len(policy)), 1, p=policy)[0]
    return action


def generate_trajectory_set(env, set_size, feature_fn, max_trajectory_length=sys.maxsize,
                            device='cpu', policy_fn=None, select_action_fn=None):
    # using select_action_fn, return action,pro
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
                if select_action_fn is not None:
                    action, action_lh = select_action_fn()
                elif policy_fn is not None:
                    try:
                        pro = policy_fn(current_state_feature)
                        action = select_action(pro)
                        action_lh = pro[action]
                    except ValueError:
                        return None, None
                next_state, reward, is_done, _ = env.step(action)
                next_state_feature = torch.tensor(feature_fn(next_state), dtype=torch.float32,
                                                  requires_grad=False).to(device)
                trajectory_i.append([current_state_feature, action, reward, next_state_feature, action_lh])
                current_state_feature = next_state_feature
                total_reward += reward
            trajectory_collection.append(trajectory_i)
    return trajectory_collection, total_reward
