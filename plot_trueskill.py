"""
从tensorboard中导出训练过程中的Trueskill变化曲线，并与top 5 AI bots 对比
Export the Trueskill variation curve during training and compare it with top 5 AI bots
"""

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def ema(data, alpha=0.8):
    """
    计算指数加权滑动平均 - calculation of Exponential Weighted Moving Average
    :param data: 输入数据，可以是列表或NumPy数组 - Input data, either a list or a NumPy array
    :param alpha: 平滑系数，范围在0到1之间 - moving factor, ranging between 0 and 1
    :return: 滑动平均结果 - result of moving average
    """
    ema_values = np.zeros_like(data)
    ema_values[0] = data[0]  # The initial value is equal to the first data point
    for i in range(1, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, required=False,
                        default=f"MicroRTSGridModeVecEnv__ppo_gridnet__1__1722344759",
                        help='Tensorboard event files or a single tensorboard '
                             'file location')
    parser.add_argument('--in-path-ais', type=str, required=False,
                        default=f"league.temp.csv",
                        help='location to save the exported data')
    args = parser.parse_args()
    in_path = f"./runs/{args.in_path}"

    print("Processing tensorboard datas...")
    event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    trueskills = dict()
    for key in keys:
        # print(key)
        if key == 'charts/trueskill':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
            raw_data = event_data.Scalars(key)
            raw_array = np.array([[x.step for x in raw_data], [x.value for x in raw_data]]).T
            sorted_indices = np.argsort(raw_array[:, 0])
            # Rearranging the entire array using a sorted index
            trueskills['agent_trained'] = raw_array[sorted_indices]
            trueskills['agent_trained'][:, 0] = trueskills['agent_trained'][:, 0] / 1e6
    print("Tensorboard data exported successfully")

    # import league.temp.csv，import trueskill from other AIs
    data = np.genfromtxt(args.in_path_ais, delimiter=',', dtype=None, encoding='utf-8')
    data = np.delete(data, 0, axis=0)
    data = np.delete(data, 11, axis=0)

    # for i in range(np.shape(data)[0]):
    for i in range(5):
        trueskills[data[i][0]] = (
            np.array([trueskills['agent_trained'][:, 0],
                      [eval(data[i][3]) for _ in range(np.shape(trueskills['agent_trained'][:, 0])[0])]]).T)

    # visualisation
    print("Drawing...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True)
    # ax.set_aspect('equal')
    # ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    # ax.set_ylim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_xlabel('Timestep in millons')
    ax.set_ylabel('Avg Trueskill')
    title_name = 'Trueskill comparision against top 5 AI bots'
    plt.title(title_name)
    color_id = 0
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for key, value in trueskills.items():
        if color_id == 0:
            ax.plot(value[:, 0], value[:, 1], '-', c=colors[color_id], alpha=0.2)
            ema_data = ema(value[:, 1], alpha=0.4)
            ax.plot(value[:, 0], ema_data, '-', c=colors[color_id], label=key + '(smooth)')
        else:
            ax.plot(value[:, 0], value[:, 1], '-', c=colors[color_id], label=key)
        color_id = color_id + 1
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(f"D:\microRTS\experiments\results\Trueskill_{args.in_path}.png")
    print(f"Plot saved to experiments/results/Trueskill_{args.in_path}.png")


if __name__ == '__main__':
    main()
