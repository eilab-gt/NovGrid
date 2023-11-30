import os
from datetime import datetime
import minigrid  # MUST BE IMPORTED TO SEE ENVIRONMENTS
from minigrid.wrappers import ImgObsWrapper
import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from novgrid.utils.parser import getparser
from novgrid.utils.novgrid_utils import make_env
from novgrid.utils.baseline_utils import MinigridCNN
from novgrid.novelty_generation.novelty_wrappers import *


def main(args):
    if args.device:
        device = th.device(args.device)
    else:
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    # Set up tracking and logging
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    if args.saves_logs == 'logs':
        defaults = getparser([])
        if defaults != args:
            logstr = ''
            for key, value in args.__dict__.items():
                if defaults.__dict__[key] != value:
                    elemstr = str(key) + '=' + str(value) + '_'
                    logstr += elemstr
    else:
        logstr = args.saves_logs


    log_dir = os.path.abspath('./logs/' + logstr + '_' + dt_string)
    os.makedirs(log_dir)

    if args.wandb_track:
        wandb_config = {
            'total_timesteps': args.total_timesteps,
            'env_name': args.env,
            'novelty_wrapper': args.novelty_wrapper,
            'novelty_episode': args.novelty_episode,
            'args': args
        }
        wandb.tensorboard.patch(root_logdir=log_dir, pytorch=True)
        wandb_run = wandb.init(
            project='novgrid_baselines',
            entity="balloch",
            settings=wandb.Settings(start_method="fork"),
            name=logstr + '_' + dt_string,
            dir='./logs/',
            config=wandb_config,
            sync_tensorboard=True,
            monitor_gym=True
        )


    env_wrappers = [ImgObsWrapper]
    wrappers_args = [{}]

    n_envs = args.num_workers

    if args.novelty_wrapper:
        novelty_wrapper = eval(args.novelty_wrapper)
        env_wrappers = [novelty_wrapper] + env_wrappers
        wrappers_args.append({})
        env_list = [make_env(env_name=args.env,
                             wrappers=env_wrappers,
                             wrapper_args=wrappers_args,
                             novelty_episode=args.novelty_episode) for _ in range(n_envs)]
        env = VecMonitor(DummyVecEnv(env_list))
    elif n_envs > 1:
        print('try make_vec_env')
        # This only works with a single wrapper for some reason.
        env = make_vec_env(args.env,
                           n_envs=n_envs,
                           seed=0,
                           wrapper_class=env_wrappers[0])
    else:
        env_list = [make_env(env_name=args.env,
                             wrappers=env_wrappers,
                             novelty_episode=args.novelty_episode) for _ in range(args.num_workers)]
        env = VecMonitor(DummyVecEnv(env_list))

    # Set up and create model
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128), 
        )

    model = PPO("CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=args.learning_rate,
                verbose=1,
                tensorboard_log=log_dir,
                device=device)
    if args.load_model:
        print(f'loading model {args.load_model}')
        model.set_parameters(args.load_model)

    # Set up experiment callbacks
    eval_callback = EvalCallback(
        VecTransposeImage(env),
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=round(args.eval_interval/n_envs),
        deterministic=True,
        render=False)
    callback_list = [eval_callback]

    if args.wandb_track:
        tracking_callback = WandbCallback(
            gradient_save_freq=10,
            model_save_path=wandb_run.dir, #'/datadrive/wandb_tmp/',
            model_save_freq=10000,
            verbose=2)
        callback_list.append(tracking_callback)
        # wandb.watch(sb_policy)

    all_callback = CallbackList(callback_list)

    # Run Experiments!
    for exp in range(args.num_exp):
        model.learn(
            total_timesteps=args.total_timesteps,
 #           log_interval=args.log_interval,
            tb_log_name='run_{}'.format(exp),
#            callback=eval_callback,
        )
        model.save(log_dir + '/' + 'run_{}'.format(exp) + '_final_model')


if __name__ == "__main__":
    config_args = getparser()
    main(config_args)
