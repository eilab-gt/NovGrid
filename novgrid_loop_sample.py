from typing import Dict, List, Any

import argparse
import gymnasium as gym
import minigrid
import json


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-name", "-e", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default="sample.json",
        help="Use the path to a json file here.",
    )
    parser.add_argument("--total-steps", "-s", type=int, default=None)
    parser.add_argument("--novelty-step", "-n", type=int, default=10)

    return parser


def make_env_list(
    env_name: str, env_configs: List[Dict[str, Any]], num_envs: int = 1
) -> List[gym.Env]:
    env_list = []

    for config in env_configs:
        if "env_name" in config:
            config = {k: config[k] for k in config if k != "env_name"}
            env_name = config["env_name"]
        env = gym.make_vec(
            env_name, num_envs=num_envs, vectorization_mode="sync", **config
        )

        env_list.append(env)

    return env_list


def run(args: argparse.Namespace) -> None:
    with open(args.config_file) as f:
        env_configs = json.load(f)
    env_list = make_env_list(env_name=args.env_name, env_configs=env_configs)

    if args.total_steps is None:
        total_steps = len(env_list) * args.novelty_step
    else:
        total_steps = args.total_steps

    env_idx = -1
    env = None

    for step_num in range(total_steps):
        if step_num % args.novelty_step == 0:
            env_idx += 1
            env = env_list[env_idx]
            env.reset()

        obs, reward, truncated, terminated, info = env.step(env.action_space.sample())

        print(
            f"env_idx: {env_idx}; step_num: {step_num}; grid_size: {env.envs[0].width}"
        )


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    run(args=args)
