import gym
import stable_baselines3.common.evaluation
from minigrid_novelty_generator.utils.parser import getparser
import sys
sys.path.append(r"C:\Users\Mustafa\code\gym-minigrid")
import gym_minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
import wandb

from wandb.integration.sb3 import WandbCallback


def main(args):
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 25000,
        "env_name": "MiniGrid-Door2Key-16x16-v0",
    }
    #    "env_name": "MiniGrid-DoorKey-16x16-v0",

    run = wandb.init(
        project="sb3_minigrid",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    def make_env():
        env = gym.make(config["env_name"])
        obs = env.reset()
        # env.render()
        env = Monitor(env)  # record stats such as returns
        env = gym_minigrid.wrappers.FlatObsWrapper(env)
        return env


    env = DummyVecEnv([make_env])
    # env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    obs = env.reset()
    eval_returns = evaluate_policy(model,
                                   env,
                                   n_eval_episodes=10,
                                   deterministic=True,
                                   )
    print(eval_returns)
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()
    run.finish()


if __name__ == "__main__":
    parser = getparser()
    args = parser.parse_args()
    main(args)
