import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def reward_action_pairs_generator(data_file_path="test.csv", k=10, size_of_sample_per_arm=100000):
    """
    experiment data generator
    Args:
        data_file_path: the data file path of the experiments
        k: how many arms(actions) you want, e.g 10-arm bandit, k=10
        size_of_sample_per_arm: the number of experiments per arm should take

    Returns:

    """
    # mean of each arm
    value_array = np.random.normal(0, 1, k)
    print(value_array)
    reward = []
    action = []
    for i in range(k):
        value = value_array[i]
        reward.append(np.random.normal(value, 10, size_of_sample_per_arm))
        action.append(np.ones(size_of_sample_per_arm) * i)
    reward = np.array(reward).reshape(1, -1)[0]
    action = np.array(action).reshape(1, -1)[0]

    dataframe = pd.DataFrame({'reward': reward, 'action': action})

    dataframe.to_csv(data_file_path, index=False, sep=',')


def depict_data_file(data_file_path='test.csv'):
    """
    depict the data file in the violin style
    Args:
        data_file_path: the path of data file

    Returns:

    """
    r_action = pd.read_csv(data_file_path)
    sns.set(style='whitegrid', color_codes=True)
    sns.violinplot(x='action', y='reward', data=r_action)
    plt.show()


if __name__ == '__main__':
    # reward_action_pairs_generator()
    depict_data_file()
