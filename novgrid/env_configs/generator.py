import abc
import os

import json
from typing import List, Any, Dict
import numpy as np


class Change(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def generate_value(self, i: int, num_tasks: int) -> Any:
        pass


class Constant(Change):

    def __init__(self, x: Any) -> None:
        self.x = x

    def generate_value(self, i: int, num_tasks: int) -> Any:
        return self.x


class IntRange(Change):

    def __init__(self, start: int, end: int, inclusive: bool = False) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.inclusive = inclusive

    def generate_value(self, i: int, num_tasks: int) -> int:
        return (
            self.start + i * (self.end + int(self.inclusive) - self.start) // num_tasks
        )


class FloatRange(Change):

    def __init__(self, start: float, end: float, inclusive: bool = False) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.inclusive = inclusive
        self._linspace_cache = {}

    def generate_value(self, i: int, num_tasks: int) -> float:
        if num_tasks not in self._linspace_cache:
            self._linspace_cache[num_tasks] = np.linspace(
                self.start, self.end, num=num_tasks, endpoint=self.inclusive
            )
        return self._linspace_cache[num_tasks][i]


class Toggle(Change):

    def __init__(self, val1: Any = False, val2: Any = True) -> None:
        super().__init__()
        self.val1 = val1
        self.val2 = val2

    def generate_value(self, i: int, num_tasks: int) -> Any:
        return self.val1 if i % 2 == 0 else self.val2


class ListChange(Change):

    def __init__(self, lst: List[Any], use_snake_boundary: bool = False) -> None:
        super().__init__()
        self.lst = lst
        if use_snake_boundary:
            self.lst = self.lst + self.lst[-2:0:-1]

    def generate_value(self, i: int, num_tasks: int) -> Any:
        return self.lst[i % len(self.lst)]


class EnvConfigGenerator:

    def __init__(self, env_id: str, num_tasks: int, changes: Dict[str, Change]) -> None:
        self.base_env_id = env_id
        self.num_tasks = num_tasks
        self.changes = changes

    def generate_env_configs(self) -> List[Dict[str, Any]]:
        return [
            {
                "env_id": self.base_env_id,
                **{
                    k: v.generate_value(i, self.num_tasks)
                    for k, v in self.changes.items()
                },
            }
            for i in range(self.num_tasks)
        ]

    def save_env_configs(self, json_file_name: str) -> List[Dict[str, Any]]:
        env_configs = self.generate_env_configs()
        with open(json_file_name, "w") as f:
            json.dump(env_configs, f)
        return env_configs

    def global_save_env_configs(
        self, name: str, override_existing_file: bool = False
    ) -> List[Dict[str, Any]]:
        fname = f"{name}.json" if ".json" not in name else name
        full_fname = os.path.join(os.path.dirname(__file__), "json", fname)
        if os.path.exists(full_fname) and not override_existing_file:
            raise ValueError(
                f"The name {name} already has a global file for its env config. To override this file use the override_existing_file flag."
            )
        self.save_env_configs(full_fname)


def test_generator_bool_toggle():
    expected_result = [
        {"env_id": "LavaGrid", "lava_on": False},
        {"env_id": "LavaGrid", "lava_on": True},
        {"env_id": "LavaGrid", "lava_on": False},
    ]

    result = EnvConfigGenerator(
        env_id="LavaGrid", num_tasks=3, changes={"lava_on": Toggle()}
    ).generate_env_configs()

    assert expected_result == result


def test_generator_int_range():
    expected_result = [
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 1},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 2},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 3},
    ]

    result = EnvConfigGenerator(
        env_id="MiniGrid-SimpleCrossingS9N0-v0",
        num_tasks=3,
        changes={"num_crossings": IntRange(1, 4)},
    ).generate_env_configs()

    assert expected_result == result


def test_generator_list_change():
    expected_result = [
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 1},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 2},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 3},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 2},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 1},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 2},
    ]

    result = EnvConfigGenerator(
        env_id="MiniGrid-SimpleCrossingS9N0-v0",
        num_tasks=6,
        changes={"num_crossings": ListChange([1, 2, 3], use_bounce_back_boundary=True)},
    ).generate_env_configs()

    assert expected_result == result


def test_generator_float_range():
    expected_result = [
        {"env_id": "CartPole", "pole_weight": 5.0},
        {"env_id": "CartPole", "pole_weight": 6.5},
        {"env_id": "CartPole", "pole_weight": 8.0},
        {"env_id": "CartPole", "pole_weight": 9.5},
    ]

    result = EnvConfigGenerator(
        env_id="CartPole",
        num_tasks=4,
        changes={"pole_weight": FloatRange(5.0, 9.5, inclusive=True)},
    ).generate_env_configs()

    assert expected_result == result


def test_generator_multi_change():
    expected_result = [
        {
            "env_id": "MiniGrid-SimpleCrossingS9N0-v0",
            "num_crossings": 1,
            "test_constant": 5,
        },
        {
            "env_id": "MiniGrid-SimpleCrossingS9N0-v0",
            "num_crossings": 2,
            "test_constant": 5,
        },
        {
            "env_id": "MiniGrid-SimpleCrossingS9N0-v0",
            "num_crossings": 3,
            "test_constant": 5,
        },
    ]

    result = EnvConfigGenerator(
        env_id="MiniGrid-SimpleCrossingS9N0-v0",
        num_tasks=3,
        changes={"num_crossings": IntRange(1, 4), "test_constant": Constant(5)},
    ).generate_env_configs()

    assert expected_result == result
