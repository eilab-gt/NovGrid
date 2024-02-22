from typing import Any, List, Optional, SupportsFloat, Tuple, Dict, Union

import os

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import json
import numpy as np
import inspect

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from novgrid.env_configs import get_env_configs
import novgrid.envs.novgrid_objects as novgrid_objects


class ListEnv(gym.Env):
    """
    A vectorized environment that chains multiple environments together.

    Attributes:
        env_lst (List[gymnasium.Env]): List of environments to chain.
        env_idx (int): Index of the current environment.
    """

    def __init__(self, env_lst: List[gym.Env]) -> None:
        """
        Initializes the ListEnv with a list of environments.

        Args:
            env_lst (List[gymnasium.Env]): List of environments to chain.
        """
        self.env_lst = env_lst
        self.env_idx = 0

    def incr_env_idx(self) -> bool:
        """
        Increments the environment index, closing the current environment and resetting to the next one.

        Returns:
            bool: True if the environment index was successfully incremented, False otherwise.
        """
        if self.env_idx >= len(self.env_lst) - 1:
            return False
        self.cur_env.close()
        self.env_idx += 1
        self.cur_env.reset()
        return True

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the current environment.

        Args:
            action (Any): Action to take.

        Returns:
            Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]: Step information.
        """
        return self.cur_env.step(action=action)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the current environment.

        Args:
            seed (Optional[int]): Seed for environment reset.
            options (Optional[Dict[str, Any]]): Additional options for reset.

        Returns:
            Tuple[Any, Dict[str, Any]]: Reset information.
        """
        return self.cur_env.reset(seed=seed, options=options)

    def render(self) -> Union[gym.core.RenderFrame, List[gym.core.RenderFrame], None]:
        """
        Renders the current environment.

        Returns:
            Union[gymnasium.core.RenderFrame, List[gymnasium.core.RenderFrame], None]: Rendered frame(s).
        """
        return self.cur_env.render()

    def close(self) -> None:
        """Closes all environments in the list."""
        for env in self.env_lst:
            env.close()

    @property
    def cur_env(self) -> gym.Env:
        """
        Gets the current environment.

        Returns:
            gymnasium.Env: Current environment.
        """
        return self.env_lst[self.env_idx]

    @property
    def unwrapped(self) -> gym.Env:
        """
        Gets the unwrapped version of the current environment.

        Returns:
            gymnasium.Env: Unwrapped current environment.
        """
        return self.cur_env

    @property
    def action_space(self) -> gym.Space:
        """
        Gets the action space of the current environment.

        Returns:
            gymnasium.Space: Action space.
        """
        return self.cur_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        """
        Gets the observation space of the current environment.

        Returns:
            gymnasium.Space: Observation space.
        """
        return self.cur_env.observation_space

    @property
    def reward_range(self) -> Tuple[SupportsFloat, SupportsFloat]:
        """
        Gets the reward range of the current environment.

        Returns:
            Tuple[SupportsFloat, SupportsFloat]: Reward range.
        """
        return self.cur_env.reward_range

    @property
    def spec(self) -> EnvSpec:
        """
        Gets the spec of the current environment.

        Returns:
            gymnasium.EnvSpec: Environment specification.
        """
        return self.cur_env.spec

    @property
    def np_random(self) -> np.random.RandomState:
        """
        Gets the random number generator of the current environment.

        Returns:
            np.random.RandomState: Random number generator.
        """
        return self.cur_env.np_random

    @property
    def render_mode(self) -> Optional[str]:
        """
        Gets the render mode of the current environment.

        Returns:
            Optional[str]: Render mode.
        """
        return self.cur_env.render_mode


class NoveltyEnv(SubprocVecEnv):
    """
    A vectorized environment with novelty injection based on specified intervals.

    Attributes:
        novelty_step (int): Number of time steps between novelty injections.
        n_envs (int): Number of environments to run in parallel.
        print_novelty_box (bool): Whether to print a novelty injection box.
        num_transfers (int): Number of transfers between environments.
        total_time_steps (int): Total time steps taken.
        last_incr (int): Time step of the last environment index increment.
        start_index (int): Starting index for environment creation.
        monitor_dir (Optional[str]): Directory for monitoring results.
    """

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
        render_mode: Optional[str] = None,
    ):
        """
        Initializes the NoveltyEnv with the provided configurations.

        Args:
            env_configs (Union[str, List[Dict[str, Any]]]): Configuration for environments.
            novelty_step (int): Number of time steps between novelty injections.
            wrappers (List[gymnasium.Wrapper]): List of wrappers to apply to each environment.
            wrapper_kwargs_lst (List[Dict[str, Any]]): List of wrapper kwargs for each wrapper.
            n_envs (int): Number of environments to run in parallel.
            seed (Optional[int]): Random seed.
            start_index (int): Starting index for environment creation.
            monitor_dir (Optional[str]): Directory for monitoring results.
            monitor_kwargs (Optional[str]): Additional kwargs for monitoring.
            start_method (Optional[str]): Start method for parallel environments.
            print_novelty_box (bool): Whether to print a novelty injection box.
            render_mode (Optional[str]): Render mode for environments.
        """
        if type(env_configs) == str:
            if os.path.exists(env_configs):
                with open(env_configs, "r") as f:
                    env_configs = json.load(f)
            else:
                env_configs = get_env_configs(env_configs)

        world_objects = {
            k.lower(): v
            for k, v in inspect.getmembers(
                novgrid_objects,
                lambda obj: inspect.isclass(obj)
                and issubclass(obj, novgrid_objects.WorldObj),
            )
        }

        for cfg in env_configs:
            for k, v in cfg.items():
                if (
                    type(v) == str
                    and v.startswith("gridobj:")
                    and v.split(":")[-1].lower() in world_objects
                ):
                    cfg[k] = world_objects[v.split(":")[-1].lower()]

        self.novelty_step = novelty_step
        self.n_envs = n_envs
        self.n_tasks = len(env_configs)
        self.print_novelty_box = print_novelty_box
        self.num_transfers = len(env_configs) - 1

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
                    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
                else:
                    env = env_id(**env_kwargs, render_mode=render_mode)

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
                return ListEnv([_make_env(config) for config in env_configs])

            return _init

        env_fns = [make_env_fn(rank=i + start_index) for i in range(n_envs)]

        if start_method is None:
            import multiprocessing as mp
            start_method = "fork" if "fork" in mp.get_all_start_methods() else None

        super().__init__(env_fns=env_fns, start_method=start_method)

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Takes a step in the parallel environments.

        Args:
            actions (np.ndarray): Actions for each environment.

        Returns:
            VecEnvStepReturn: The observations, rewards, dones, and infos from each environment
        """
        observations, rewards, dones, infos = super().step(actions)
        # Increment total time steps
        self.total_time_steps += self.n_envs
        if self.total_time_steps - self.last_incr > self.novelty_step:
            self.last_incr = self.total_time_steps
            # Trigger the novelty if enough steps have passed
            novelty_injected = self.env_method("incr_env_idx")
            dones[:] = True

            if np.any(novelty_injected) and self.print_novelty_box:
                s = f"| Novelty Injected (on env {self.get_attr('env_idx')}) |"
                print("-" * len(s))
                print(s)
                print("-" * len(s))

        return observations, rewards, dones, infos
