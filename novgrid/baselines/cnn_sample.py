import os
from datetime import datetime
import gym_minigrid  # MUST BE IMPORTED TO SEE ENVIRONMENTS
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
from novgrid.utils.baseline_utils import MinigridCNN
from novgrid.novelty_generation.novelty_wrappers import *

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


def main(args):
    # Set up tracking and logging
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.abspath('./logs/' + args.saves_logs + '_' + dt_string)
    wandb_root = os.path.abspath('./logs/')
    wandb_log = os.path.join(wandb_root, args.saves_logs + '_' + dt_string)
    if args.wandb_track:
        wandb_config = {
            "total_timesteps": args.total_timesteps,
            "env_name": args.env,
        }
        run = wandb.init(
            project="sb3_minigrid",
            dir=wandb_root,
            config=wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # Create environments
    # env_wrappers = [DoorKeyChange]
    env_wrappers = [DoorKeyNoveltyWrapper]
    env_list = [make_env(args.env, log_dir, wrappers=env_wrappers) for _ in range(args.num_workers)]
    env = VecMonitor(DummyVecEnv(env_list), filename=log_dir)
    # env = DummyVecEnv([lambda: Monitor(CustomEnv(reward_func=FUNCTION), log_dir, allow_early_resets=True) for _ in range(num_cpu)])

    # Set up and create model
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128), )
    model = PPO("CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                verbose=1,
                tensorboard_log=log_dir,
                device=device)
    # if args.load_model:
    #     print(f'loading model {args.load_model}')
    #     model.set_parameters(args.load_model)

    # Set up experiment callbacks
    eval_callback = EvalCallback(
        VecTransposeImage(env),
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False)
    callback_list = [eval_callback]
    if args.wandb_track:
        tracking_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_path=wandb_log,
            model_save_freq=10000,
            verbose=0)
        callback_list.append(tracking_callback)
    all_callback = CallbackList(callback_list)

    # Run Experiments!
    # TODO: this way of having multiple runs probably will not work with wandb, it will think its all one exp
    for exp in range(args.num_exp):
        model.learn(
            total_timesteps=10000000,
            tb_log_name='run_{}'.format(exp),
            callback=all_callback,
        )
        model.save(log_dir + '/' + 'run_{}'.format(exp) + '_final_model')
    if args.wandb_track:
        run.finish()


if __name__ == "__main__":
    config_args = getparser()
    main(config_args)
