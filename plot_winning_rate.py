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
    ema_values[0] = data[0]  # The initial value is set to the first index
    for i in range(1, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path1', type=str, required=False,
                        # default=f"MicroRTSGridModeVecEnv__ppo_gridnet_eval_winorlose__1__2024-08-09 11:24:47.738407",
                        default=f"MicroRTSGridModeVecEnv__ppo_gridnet_eval_winorlose__1__2024-08-09 08:30:27.693225",
                        help='Tensorboard event files or a single tensorboard '
                             'file location')
    parser.add_argument('--in-path2', type=str, required=False,
                        # default=f"MicroRTSGridModeVecEnv__ppo_gridnet_eval_winorlose__1__2024-08-10 12:40:48.764367",
                        default=f"MicroRTSGridModeVecEnv__ppo_gridnet_eval_winorlose__1__2024-08-11 08:56:21.812618",
                        help='Tensorboard event files or a single tensorboard '
                             'file location')
    parser.add_argument('--in-path-ais', type=str, required=False,
                        default=f"league.temp.csv",
                        help='location to save the exported data')
    args = parser.parse_args()

    all_ais = [
        "randomBiasedAI",
        "randomAI",
        "passiveAI",
        "workerRushAI",
        "lightRushAI",
        "coacAI",
        "naiveMCTSAI",
        "mixedBot",
        "rojo",
        "izanagi",
        "tiamat",
        "droplet",
        "guidedRojoA3N"
    ]

    print("Processing tensorboard datas...")
    # group 1
    in_path1 = f"./runs/{args.in_path1}"
    event_data = event_accumulator.EventAccumulator(in_path1)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    winning_rate = dict()
    for ais in all_ais:
        win_games = -1
        lose_games = -1
        tie_games = -1
        for key in keys:
            if key == 'charts/' + ais + '/win':
                win_games = event_data.Scalars(key)[0].value
            if key == 'charts/' + ais + '/loss':
                lose_games = event_data.Scalars(key)[0].value
            if key == 'charts/' + ais + '/tie':
                tie_games = event_data.Scalars(key)[0].value
        if win_games != -1 and lose_games != -1 and tie_games != -1:
            winning_rate[ais] = win_games / (win_games + lose_games + tie_games) * 100

    # group 2
    in_path2 = f"./runs/{args.in_path2}"
    event_data = event_accumulator.EventAccumulator(in_path2)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    winning_rate1 = dict()
    for ais in all_ais:
        win_games = -1
        lose_games = -1
        tie_games = -1
        for key in keys:
            if key == 'charts/' + ais + '/win':
                win_games = event_data.Scalars(key)[0].value
            if key == 'charts/' + ais + '/loss':
                lose_games = event_data.Scalars(key)[0].value
            if key == 'charts/' + ais + '/tie':
                tie_games = event_data.Scalars(key)[0].value
        if win_games != -1 and lose_games != -1 and tie_games != -1:
            winning_rate1[ais] = win_games / (win_games + lose_games + tie_games) * 100

    print("Tensorboard data exported successfully")
    # visualisation
    print("Drawing...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(False)
    # ax.set_aspect('equal')
    # ax.set_xlim([-np.max(task_locations) * 1.4, np.max(task_locations) * 1.4])
    ax.set_ylim([0, 110])
    # ax.set_xlabel('Timestep in millons')
    ax.set_ylabel('Winning Rate %')

    # title_name = 'Comparison of Winning Rate for Sparse Rewards and Shaped Rewards'
    # title_name = 'Comparison of Winning Rate for Curriculum Learning and Only CoacAI'
    # title_name = 'Comparison of Winning Rate for Selfplay and Diverse Opponents'
    # title_name = 'Comparison of Winning Rate for Different Weights of Combat Unit'
    title_name = 'Comparison of Winning Rate for Different Weights of Construct Building'
    plt.title(title_name)

    index = range(len(all_ais))

    # ax.bar(index, winning_rate.values(), label='Shaped Rewards', width=0.3)
    # ax.bar(index, winning_rate.values(), label='Only CoacAI', width=0.3)
    # ax.bar(index, winning_rate.values(), label='Diverse Opponents', width=0.3)
    # ax.bar(index, winning_rate.values(), label='CombatUnit 4.0', width=0.3)
    ax.bar(index, winning_rate.values(), label='ConstructBuilding 0.2', width=0.3)

    # ax.bar([i + 0.3 for i in index], winning_rate1.values(), label='Sparse Rewards', width=0.3)
    # ax.bar([i + 0.3 for i in index], winning_rate1.values(), label='Curriculum Learning', width=0.3)
    # ax.bar([i + 0.3 for i in index], winning_rate1.values(), label='Selfplay', width=0.3)
    # ax.bar([i + 0.3 for i in index], winning_rate1.values(), label='CombatUnit 8.0', width=0.3)
    ax.bar([i + 0.3 for i in index], winning_rate1.values(), label='ConstructBuilding 0.5', width=0.3)

    ax.set_xticks([i + 0.3 / 2 for i in index])
    ax.set_xticklabels(all_ais, rotation=45)
    fig.tight_layout()
    plt.legend(loc='upper left')
    # plt.show()

    # filename = 'WinningRate_Exp1.png'
    # filename = 'WinningRate_Exp2.png'
    # filename = 'WinningRate_Exp4_2.png'
    filename = 'WinningRate_Exp4_3.png'
    plt.savefig(f"/home/dayday/Reinforcement Learning/MicroRTS-Py/experiments/results/{filename}")
    print(f"Plot saved to experiments/results/{filename}")


if __name__ == '__main__':
    main()
