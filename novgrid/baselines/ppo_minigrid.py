import os
from datetime import datetime

import gym_minigrid  # MUST BE IMPORTED TO SEE ENVIRONMENTS
from gym_minigrid.wrappers import FlatObsWrapper
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO

from novgrid.utils.parser import getparser
from novgrid.utils.novgrid_utils import make_env
from novgrid.novelty_generation.novelty_wrappers import *

device = th.device('cuda' if th.cuda.is_available() else 'cpu')


def main(args):
    # Set up tracking
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.abspath('./logs/' + args.saves_logs + '_' + dt_string)
    os.makedirs(log_dir)

    # Create environments
    novelty_wrapper = eval(args.novelty_wrapper)
    env_wrappers = [novelty_wrapper, FlatObsWrapper]
    env_list = [make_env(args.env, log_dir, env_wrappers, args.novelty_episode) for _ in range(args.num_workers)]
    env = VecMonitor(DummyVecEnv(env_list))

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

    for exp in range(args.num_exp):
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name='run_{}'.format(exp)
        )
        model.save(log_dir + '/' + 'run_{}'.format(exp) + '_final_model')


if __name__ == "__main__":
    config_args = getparser()
    main(config_args)
