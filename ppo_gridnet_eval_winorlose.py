# http://proceedings.mlr.press/v97/han19a/han19a.pdf
"""
- 评估训练好的Agent，与13个AI bots对局，统计输赢情况
- Evaluate the trained Agent, let it play against 13 AI bots, calculate wins and loses
- 结果保存至tensorboard，可以查看 - Save the results to the tensorboard
- 在本地/experiments/results生成对局的png - Generate a png of the game in local/experiments/results
- 为后续对比胜率plot_winning_rate.py提供数据 - Provide data for subsequent comparison of winning rate plot_winning_rate.py
"""

import argparse
import os
import random
import time
import datetime
from distutils.util import strtobool
import matplotlib.pyplot as plt

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter
import glob

from gym_microrts import microrts_ai  # noqa


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25600,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--num-eval-runs', type=int, default=1,
                        help='the number of bot game environment; 16 bot envs measn 16 games')

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    # parser.add_argument("--agent-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
    #     help="the path to the agent's model")
    parser.add_argument("--agent-model-path", type=str,
                        default="models/MicroRTSGridModeVecEnv__ppo_gridnet__1__1722439278/agent.pt",
                        help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str,
                        default="models/MicroRTSGridModeVecEnv__ppo_gridnet__1__1722238308/agent.pt",
                        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="randomBiasedAI",
                        help='the opponent AI to evaluate against')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet", choices=["ppo_gridnet", "ppo_gridnet"],
                        help='the output path of the leaderboard csv')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.num_bot_envs, args.num_selfplay_envs = 1, 0
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == "ppo_gridnet":
        from ppo_gridnet import Agent, MicroRTSStatsRecorder

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    else:
        from ppo_gridnet import Agent, MicroRTSStatsRecorder

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.datetime.fromtimestamp(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        CHECKPOINT_FREQUENCY = 10
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    all_ais = {
        "randomBiasedAI": microrts_ai.randomBiasedAI,
        "randomAI": microrts_ai.randomAI,
        "passiveAI": microrts_ai.passiveAI,
        "workerRushAI": microrts_ai.workerRushAI,
        "lightRushAI": microrts_ai.lightRushAI,
        "coacAI": microrts_ai.coacAI,
        "naiveMCTSAI": microrts_ai.naiveMCTSAI,
        "mixedBot": microrts_ai.mixedBot,
        "rojo": microrts_ai.rojo,
        "izanagi": microrts_ai.izanagi,
        "tiamat": microrts_ai.tiamat,
        "droplet": microrts_ai.droplet,
        "guidedRojoA3N": microrts_ai.guidedRojoA3N
    }
    # all_ais = {
    #     "randomBiasedAI": microrts_ai.randomBiasedAI,
    #     "randomAI": microrts_ai.randomAI
    # }
    ai_names, ais_values = list(all_ais.keys()), list(all_ais.values())
    ai_match_stats = dict(zip(ai_names, np.zeros((len(ais_values), 3))))
    args.num_envs = len(ais_values)
    ai_envs = []

    for i in range(len(ais_values)):
        ais = [ais_values[i]]
        envs = MicroRTSGridModeVecEnv(
            num_bot_envs=len(ais),
            num_selfplay_envs=args.num_selfplay_envs,
            partial_obs=args.partial_obs,
            max_steps=5000,
            render_theme=2,
            ai2s=ais,
            map_paths=["maps/16x16/basesWorkers16x16.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        )
        envs = MicroRTSStatsRecorder(envs)
        envs = VecMonitor(envs)
        if args.capture_video:
            envs = VecVideoRecorder(
                envs, f"videos/{experiment_name}/{ai_names[i]}",
                record_video_trigger=lambda x: x % 4000 == 0, video_length=int(1e10))
        ai_envs += [envs]
    # assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
    ## CRASH AND RESUME LOGIC:
    agent.load_state_dict(torch.load(args.agent_model_path, map_location=device))
    agent.eval()
    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    for i, envs in enumerate(ai_envs):
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        game_count = 0

        while True:
            # envs.render()
            with torch.no_grad():
                invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)

                action, logproba, _, _, vs = agent.get_action_and_value(
                    next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                )

            try:
                next_obs, rs, ds, infos = envs.step(action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    print("against", ai_names[i], info['microrts_stats']['WinLossRewardFunction'])
                    if info['microrts_stats']['WinLossRewardFunction'] == -1.0:
                        ai_match_stats[ai_names[i]][0] += 1
                    elif info['microrts_stats']['WinLossRewardFunction'] == 0.0:
                        ai_match_stats[ai_names[i]][1] += 1
                    elif info['microrts_stats']['WinLossRewardFunction'] == 1.0:
                        ai_match_stats[ai_names[i]][2] += 1
                    game_count += 1
            if game_count >= args.num_eval_runs:
                envs.close()
                for (label, val) in zip(["loss", "tie", "win"], ai_match_stats[ai_names[i]]):
                    writer.add_scalar(f"charts/{ai_names[i]}/{label}", val, 0)
                # if args.capture_video:
                #     video_files = glob.glob(f'videos/{experiment_name}/{ai_names[i]}/*.mp4')
                    # for video_file in video_files:
                    #     print(video_file)
                    #     wandb.log({f"RL agent against {ai_names[i]}": wandb.Video(video_file)})
                    # labels, values = ["loss", "tie", "win"], ai_match_stats[ai_names[i]]
                    # data = [[label, val] for (label, val) in zip(labels, values)]
                    # table = wandb.Table(data=data, columns = ["match result", "number of games"])
                    # wandb.log({ai_names[i]: wandb.plot.bar(table, "match result", "number of games", title=f"RL agent against {ai_names[i]}")})
                break
    # envs.close()

    n_rows, n_cols = 3, 5
    fig = plt.figure(figsize=(5 * 3, 4 * 3))
    for i, var_name in enumerate(ai_names):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.bar(["loss", "tie", "win"], ai_match_stats[var_name])
        ax.set_title(var_name)
    fig.suptitle(args.agent_model_path)
    fig.tight_layout()
    cumulative_match_results = np.array(list(ai_match_stats.values())).sum(0)
    cumulative_match_results_rate = cumulative_match_results / cumulative_match_results.sum()
    # if args.prod_mode:
        # wandb.log({"Match results": wandb.Image(fig)})
    for (label, val) in zip(["loss", "tie", "win"], cumulative_match_results):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    for (label, val) in zip(["loss rate", "tie rate", "win rate"], cumulative_match_results_rate):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    plt.savefig(f"results/{experiment_name}.png")

    writer.close()
