import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.ticker import FuncFormatter

def extract_game_name(folder_name):
    match = re.match(r'ALE-(.+)-v5_\d+-\d+-\d+-\d+-\d+', folder_name)
    if match:
        return match.group(1)
    return None


def load_tensorboard_data(log_dir):
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    # 提取reward和q值的数据
    rewards = [scalar.value for scalar in accumulator.Scalars('avg_reward')]
    q_values = [scalar.value for scalar in accumulator.Scalars('q')]

    return rewards, q_values

def formatter(x, pos):
    return f'{int(x / 10)}M'  # Assumes x is the index, every index is 100000 updates

def plot_data(game_data):
    plt.rcParams.update({'font.size': 14})  # 设置字体大小适合学术论文

    for game_name, data in game_data.items():
        rewards, q_values = zip(*data)

        min_length = min(len(q) for q in q_values)
        q_values = [q[:min_length] for q in q_values]

        reward_means = np.mean(rewards, axis=0)
        reward_medians = np.median(rewards, axis=0)
        reward_max = np.max(rewards, axis=0)
        reward_min = np.min(rewards, axis=0)

        q_means = np.mean(q_values, axis=0)[::10]
        q_medians = np.median(q_values, axis=0)[::10]
        q_max = np.max(q_values, axis=0)[::10]
        q_min = np.min(q_values, axis=0)[::10]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=300)  # 使用更高的DPI确保图像质量

        axs[0].plot(reward_means, label='Mean Reward')
        axs[0].plot(reward_medians, linestyle='dashed', label='Median Reward')
        axs[0].fill_between(range(len(reward_means)), reward_min, reward_max, color='gray', alpha=0.5)
        axs[0].set_title(f'Reward Statistics for {game_name}')
        axs[0].set_xlabel('Steps (in thousands)')
        axs[0].set_ylabel('Reward')
        axs[0].legend()

        axs[1].plot(q_means, label='Mean Q')
        axs[1].plot(q_medians, linestyle='dashed', label='Median Q')
        axs[1].fill_between(range(len(q_means)), q_min, q_max, color='gray', alpha=0.5)
        axs[1].set_title(f'Q Value Statistics for {game_name}')
        axs[1].xaxis.set_major_formatter(FuncFormatter(formatter))
        axs[1].set_xlabel('Updates (in millions)')
        axs[1].set_ylabel('Q value')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f'./{game_name}_stats.png')  # 保存为PNG格式


# 主执行函数
def main(log_dir):
    game_data = {}
    for folder in os.listdir(log_dir):
        if '.log' in folder:
            continue
        game_name = extract_game_name(folder)
        if game_name:
            data_path = os.path.join(log_dir, folder)
            rewards, q_values = load_tensorboard_data(data_path)
            if len(rewards) != 100:
                continue
            if game_name not in game_data:
                game_data[game_name] = []
            game_data[game_name].append((rewards, q_values))

    plot_data(game_data)


# 示例调用
main('../exps/dqn')
