from typing import Any, Callable, List, Optional, SupportsFloat, Tuple, Dict, Union

import os

import argparse
import gymnasium as gym
import json
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

import minigrid


class ListEnv(gym.Env):
    def __init__(self, env_lst: List[gym.Env]) -> None:
        self.env_lst = env_lst
        self.env_idx = 0

    def incr_env_idx(self) -> bool:
        if self.env_idx >= len(self.env_lst) - 1:
            return False
        self.cur_env.close()
        self.env_idx += 1
        self.cur_env.reset()
        return True

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        return self.cur_env.step(action=action)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        return self.cur_env.reset(seed=seed, options=options)

    def render(self) -> Union[gym.core.RenderFrame, List[gym.core.RenderFrame], None]:
        return self.cur_env.render()

    def close(self):
        for env in self.env_lst:
            env.close()

    @property
    def cur_env(self):
        return self.env_lst[self.env_idx]

    @property
    def unwrapped(self):
        return self.cur_env

    @property
    def action_space(self):
        return self.cur_env.action_space

    @property
    def observation_space(self):
        return self.cur_env.observation_space

    @property
    def reward_range(self):
        return self.cur_env.reward_range

    @property
    def spec(self):
        return self.cur_env.spec

    @property
    def np_random(self):
        return self.cur_env.np_random

    @property
    def width(self):
        return self.cur_env.width

    @property
    def height(self):
        return self.cur_env.height


class NoveltyEnv(SubprocVecEnv):
    def __init__(
        self,
        env_configs: Union[str, List[Dict[str, Any]]],
        novelty_step: int,
        wrappers: List[gym.Wrapper] = [],
        wrapper_kwargs_lst: List[Dict[str, Any]] = [],
        n_envs: int = 1,
        seed: Optional[int] = None,
        start_index: int = 0,
        monitor_dir: Optional[str] = None,
        monitor_kwargs: Optional[str] = None,
        start_method: Optional[str] = None,
        print_novelty_box: bool = False,
    ):
        if type(env_configs) == str:
            with open(env_configs, "r") as f:
                self.env_configs = json.load(f)
        else:
            self.env_configs = env_configs

        self.novelty_step = novelty_step
        self.n_envs = n_envs
        self.print_novelty_box = print_novelty_box
        self.num_transfers = len(self.env_configs) - 1

        self.total_time_steps = 0
        self.last_incr = 0

        self.start_index = start_index
        self.monitor_dir = monitor_dir
        monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs

        def make_env_fn(rank):
            def _make_env(config):
                env_id = config["env_id"]
                env_kwargs = {k: v for k, v in config.items() if k != "env_id"}

                # Initialize the environment
                if isinstance(env_id, str):
                    env = gym.make(env_id, **env_kwargs)
                else:
                    env = env_id(**env_kwargs)

                # Optionally use the random seed provided
                if seed is not None:
                    env.seed(seed + rank)
                    env.action_space.seed(seed + rank)

                # Wrap the env in a Monitor wrapper
                # to have additional training information
                monitor_path = (
                    os.path.join(monitor_dir, str(rank))
                    if monitor_dir is not None
                    else None
                )
                # Create the monitor folder if needed
                if monitor_path is not None:
                    os.makedirs(monitor_path, exist_ok=True)
                env = Monitor(env, filename=monitor_path, **monitor_kwargs)

                # Wrap the environment with the provided wrappers
                for wrapper_cls, wrapper_kwargs in zip(
                    wrappers,
                    wrapper_kwargs_lst
                    + [{}] * max(0, len(wrappers) - len(wrapper_kwargs_lst)),
                ):
                    env = wrapper_cls(env, **wrapper_kwargs)

                return env

            def _init():
                # Returns a list env with each env constructed from the config in env_configs
                return ListEnv([_make_env(config) for config in self.env_configs])

            return _init

        env_fns = [make_env_fn(rank=i + start_index) for i in range(n_envs)]

        super().__init__(env_fns=env_fns, start_method=start_method)

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        result = super().step(actions)
        # Increment total time steps
        self.total_time_steps += self.n_envs
        if self.total_time_steps - self.last_incr > self.novelty_step:
            self.last_incr = self.total_time_steps
            # Trigger the novelty if enough steps have passed
            novelty_injected = self.env_method("incr_env_idx")
            if np.any(novelty_injected) and self.print_novelty_box:
                s = f"| Novelty Injected (on env {self.get_attr('env_idx')}) |"
                print("-" * len(s))
                print(s)
                print("-" * len(s))

        return result


CONFIG_FILE = "sample2.json"
TOTAL_TIME_STEPS = None
NOVELTY_STEP = 10
N_ENVS = 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        default=CONFIG_FILE,
        help="Use the path to a json file here.",
    )
    parser.add_argument("--total-time-steps", "-t", type=int, default=TOTAL_TIME_STEPS)
    parser.add_argument("--novelty-step", "-n", type=int, default=NOVELTY_STEP)
    parser.add_argument("--n-envs", "-e", type=int, default=N_ENVS)

    return parser


def run_test(
    args: argparse.Namespace,
):
    env = NoveltyEnv(
        env_configs=args.config_file,
        novelty_step=args.novelty_step,
        n_envs=args.n_envs,
    )

    env.reset()

    if args.total_time_steps is None:
        total_time_steps = (env.num_transfers + 1) * args.novelty_step
    else:
        total_time_steps = args.total_time_steps

    for step_num in range(total_time_steps):
        observations, rewards, dones, infos = env.step(
            [env.action_space.sample() for _ in range(args.n_envs)]
        )
        print(
            f"step_num: {step_num}; env_idx: {env.get_attr('env_idx')}; size: {env.get_attr('width')}"
        )


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    run_test(args=args)
