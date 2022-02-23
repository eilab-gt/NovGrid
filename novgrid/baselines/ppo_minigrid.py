import imp
import os
from datetime import datetime
import sys

import gym_minigrid  # MUST BE IMPORTED TO SEE ENVIRONMENTS
from gym_minigrid.wrappers import FlatObsWrapper
import torch as th
import gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from novgrid.utils.parser import getparser
from novgrid.utils.novgrid_utils import make_env
from novgrid.novelty_generation.novelty_wrappers import *

import matplotlib.pyplot as plt

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


def main(args):
    # Set up tracking
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.abspath('./logs/' + args.saves_logs + '_' + dt_string)
    wandb_root = os.path.abspath('./logs/')
    wandb_log = os.path.join(wandb_root, args.saves_logs + '_' + dt_string)

    # Create environments
    env_wrappers = [GoalLocationChange]
    env_list = [make_env(args.env, log_dir, env_wrappers, args.novelty_episode) for _ in range(args.num_workers)]
    env = VecMonitor(DummyVecEnv(env_list))
    # env = DummyVecEnv([lambda: Monitor(CustomEnv(reward_func=FUNCTION), log_dir, allow_early_resets=True) for _ in range(num_cpu)])

    # Set up logging
    if args.wandb_track:
        wandb_config = {
            "total_timesteps": args.total_timesteps,
            "env_name": args.env,
            "novelty_episode": args.novelty_episode,
            "novelty_wrappers": env_wrappers[0].__name__
        }
        run = wandb.init(
            project="minigrid",
            dir=wandb_root,
            config=wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb.run.name = env_wrappers[0].__name__

    # Set up and create model
    model = PPO("MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                verbose=1,
                tensorboard_log=log_dir,
                device=device)
    if args.load_model:
        print(f'loading model {args.load_model}')
        model.set_parameters(args.load_model)

    # Run Experiments!
    # TODO: this way of having multiple runs probably will not work with wandb, it will think its all one exp

    for exp in range(args.num_exp):
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name='run_{}'.format(exp)
        )
        model.save(log_dir + '/' + 'run_{}'.format(exp) + '_final_model')
    if args.wandb_track:
        run.finish()


if __name__ == "__main__":
    config_args = getparser()
    main(config_args)
