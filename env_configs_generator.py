import abc

import json
from typing import List, Union, Optional, Any, Dict
import numpy as np


class Change(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def generate_value(self, i: int, num_tasks: int):
        pass


class Constant(Change):

    def __init__(self, x: Any) -> None:
        self.x = x

    def generate_value(self, i: int, num_tasks: int):
        return self.x


class IntRange(Change):

    def __init__(self, start: int, end: int, inclusive: bool = False) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.inclusive = inclusive

    def generate_value(self, i: int, num_tasks: int):
        return (
            self.start + i * (self.end + int(self.inclusive) - self.start) // num_tasks
        )


class FloatRange(Change):

    def __init__(self, start: float, end: float, inclusive: bool = False) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.inclusive = inclusive
        self.__linspace_cache = {}

    def generate_value(self, i: int, num_tasks: int):
        if num_tasks not in self.__linspace_cache:
            self.__linspace_cache[num_tasks] = np.linspace(
                self.start, self.end, num=num_tasks, endpoint=self.inclusive
            )
        return self.__linspace_cache[num_tasks]


class Toggle(Change):

    def __init__(self, val1: Any = False, val2: Any = True) -> None:
        super().__init__()
        self.val1 = val1
        self.val2 = val2

    def generate_value(self, i: int, num_tasks: int):
        return self.val1 if i % 2 == 0 else self.val2


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


def generate_config_json(
    base_env: str,
    num_tasks: int = 0,
    change_vars: Optional[List[str]] = None,
    change_types: Optional[List[type]] = None,
    change_ranges: Optional[List[Union[tuple, None]]] = None,
):
    """
    base_env : str, the name of the env_id
    num_tasks : the number of changes (so the number of resulting environments is cahnge_count+1),
    change_vars : the list of kwarg variable names to change,
    change_types : the list of types of the kwarg variables to change, options 'bool', 'int', or 'float'
    change_ranges : the ranges of the kwarg variables to change if type is not bool.
                    Each must be len == 2 or None for bool.
                    For an int will return the largest subinterval divisible by num_taskss.
    """
    if change_ranges is not None:
        for r in change_ranges:
            assert r is None or len(r) == 2

    json_data = []
    # for n in range(num_tasks):
    var_values = {}
    for idx, var in enumerate(change_vars):
        if change_types[idx] is bool:
            var_values[var] = [bool(i % 2) for i in range(num_tasks)]
        elif change_types[idx] is int:
            var_values[var] = [
                val * (change_ranges[idx][1] - change_ranges[idx][0]) // num_tasks
                + change_ranges[idx][0]
                for val in range(num_tasks)
            ]
        elif change_types[idx] is float:
            var_values[var] = list(np.linspace(*change_ranges[idx]), num_tasks)
        else:
            raise TypeError
    for i in range(num_tasks):
        json_data.append(
            {
                "env_id": base_env,
                **dict(map(lambda x: (x[0], x[1][i]), var_values.items())),
            }
        )
    return json_data


def assert_value(ground_truth, value):
    try:
        assert ground_truth == value
    except:
        print("Test Failed!")
        print("Expected:", ground_truth)
        print("Received:", value)


def test1():
    case1 = [
        {"env_id": "LavaGrid", "lava_on": False},
        {"env_id": "LavaGrid", "lava_on": True},
        {"env_id": "LavaGrid", "lava_on": False},
    ]

    case2 = [
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 1},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 2},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 3},
    ]

    assert_value(case1, generate_config_json("LavaGrid", 3, ["lava_on"], [bool]))
    assert_value(
        case2,
        generate_config_json(
            "MiniGrid-SimpleCrossingS9N0-v0", 3, ["num_crossings"], [int], [(1, 4)]
        ),
    )


def test2():
    case1 = [
        {"env_id": "LavaGrid", "lava_on": False},
        {"env_id": "LavaGrid", "lava_on": True},
        {"env_id": "LavaGrid", "lava_on": False},
    ]

    case2 = [
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 1},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 2},
        {"env_id": "MiniGrid-SimpleCrossingS9N0-v0", "num_crossings": 3},
    ]

    assert_value(
        case1,
        EnvConfigGenerator(
            env_id="LavaGrid", num_tasks=3, changes={"lava_on": Toggle()}
        ).generate_env_configs(),
    )
    assert_value(
        case2,
        EnvConfigGenerator(
            env_id="MiniGrid-SimpleCrossingS9N0-v0",
            num_tasks=3,
            changes={"num_crossings": IntRange(1, 4)},
        ).generate_env_configs(),
    )


if __name__ == "__main__":
    test1()
    test2()
