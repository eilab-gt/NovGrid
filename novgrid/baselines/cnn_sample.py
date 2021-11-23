import os
from datetime import datetime
import torch as th
from torch import nn
import gym
import sys
import gym_minigrid
import argparse

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import wandb

from wandb.integration.sb3 import WandbCallback



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='models/best_model.zip')
    parser.add_argument('-t','--total_timesteps', type=int, default=500000)
    parser.add_argument('-e', '--env', type=str, default='MiniGrid-DoorKey-6x6-v0') 
    parser.add_argument('-s', '--saves_logs', type=str, default='minigrid_cnn_logs')
    parser.add_argument('--num_exp', type=int, default=1)
    # parser.add_argument('--debug', type=bool, default=False, action='store_true')
    return parser.parse_args()


class MinigridCNN(BaseFeaturesExtractor):
    """
    CNN for minigrid:
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        final_dim = 64
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, final_dim, (2, 2)),
            nn.ReLU(),
            nn.Flatten())
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
 	
        #n = observation_space.shape[1]
        #m = observation_space.shape[2]
        #self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*final_dim
        #self.linear = nn.Sequential(nn.Linear(self.image_embedding_size, features_dim), nn.ReLU())
 
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
 
 
def make_env(config, log_dir):
    '''
    I think that you have to have this function because the vectorization code expects a function
    '''
    def _init():
        env = gym.make(config["env_name"])
        env = gym_minigrid.wrappers.ImgObsWrapper(env)
        env = Monitor(env, log_dir)
        # obs = env.reset()
        return env
    return _init

     
def main(args):
  
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    config = {
        "total_timesteps": 10000000,
        "env_name": "MiniGrid-DoorKey-6x6-v0",
    }
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.abspath('./logs/' + args.saves_logs + '_' + dt_string)
    wandb_root = os.path.abspath('./logs/')
    wandb_log = os.path.join(wandb_root, args.saves_logs + '_' + dt_string)
    # env = DummyVecEnv([lambda: Monitor(CustomEnv(reward_func=FUNCTION), log_dir, allow_early_resets=True) for _ in range(num_cpu)])

    run = wandb.init(
        project="sb3_minigrid",
        dir=wandb_root,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    env = VecMonitor(DummyVecEnv([make_env(config, log_dir)]), filename=log_dir)
    #print(env.observation_space.shape)
    
    eval_callback = EvalCallback(VecTransposeImage(env), best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    tracking_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=wandb_log,
        model_save_freq=10000,
        verbose=2)

    all_callback = CallbackList([tracking_callback, eval_callback])

    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
        )

    model = PPO("CnnPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log=str(args.saves_logs), 
                device=device)

    if args.load:
        print(f'loading model{args.load}')
        # params = model_1.get_parameters()
        model.set_parameters(args.load)
    
    for exp in range(args.num_exp):
        model.learn(
            total_timesteps=args.total_timesteps, 
            tb_log_name='run_{}'.format(exp), 
            callback=all_callback,
        )
        model.save(log_dir+'/'+'run_{}'.format(exp)+'_final_model')
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
