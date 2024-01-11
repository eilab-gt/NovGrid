import argparse
import gymnasium as gym
import minigrid
import json


parser = argparse.ArgumentParser()

parser.add_argument("--env-name", "-e", type=str, default="MiniGrid-Empty-8x8-v0")
parser.add_argument(
    "--config-file",
    "-c",
    type=str,
    default="sample.json",
    help="Use a json file for multiple configs.",
)
parser.add_argument("--total-steps", "-s", type=int, default=None)


def make_env_list(env_name, env_configs, num_envs=1):
    env_list = []

    for config in env_configs:
        env = gym.make_vec(
            env_name, num_envs=num_envs, vectorization_mode="sync", **config
        )

        env_list.append(env)

    return env_list


def run(args):
    with open(args.config_file) as f:
        env_configs = json.load(f)
    env_list = make_env_list(env_name=args.env_name, env_configs=env_configs)

    novelty_step = 10

    if args.total_steps is None:
        total_steps = len(env_list) * novelty_step
    else:
        total_steps = args.total_steps

    env_idx = -1
    env = None

    for step_num in range(total_steps):
        if step_num % novelty_step == 0:
            env_idx += 1
            env = env_list[env_idx]
            env.reset()

        obs, reward, truncated, terminated, info = env.step(env.action_space.sample())

        print(f"env_idx: {env_idx}; step_num: {step_num}; grid_size: {env.envs[0].width}")


if __name__ == "__main__":
    args = parser.parse_args()

    run(args=args)
