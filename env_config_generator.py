import json
from typing import List, Union, Optional
import numpy as np


def generate_config_json(
    base_env: str,
    num_envs: int = 0,
    change_vars: Optional[List[str]] = None,
    change_types: Optional[List[type]] = None,
    change_ranges: Optional[List[Union[tuple, None]]] = None,
):
    """
    base_env : str, the name of the env_id
    num_envs : the number of changes (so the number of resulting environments is cahnge_count+1),
    change_vars : the list of kwarg variable names to change,
    change_types : the list of types of the kwarg variables to change, options 'bool', 'int', or 'float'
    change_ranges : the ranges of the kwarg variables to change if type is not bool.
                    Each must be len == 2 or None for bool.
                    For an int will return the largest subinterval divisible by num_envss.
    """
    if change_ranges is not None:
        for r in change_ranges:
            assert r is None or len(r) == 2

    json_data = []
    # for n in range(num_envs):
    var_values = {}
    for idx, var in enumerate(change_vars):
        if change_types[idx] is bool:
            var_values[var] = [bool(i % 2) for i in range(num_envs)]
        elif change_types[idx] is int:
            var_values[var] = [
                val * (change_ranges[idx][1] - change_ranges[idx][0]) // num_envs
                + change_ranges[idx][0]
                for val in range(num_envs)
            ]
        elif change_types[idx] is float:
            var_values[var] = list(np.linspace(*change_ranges[idx]), num_envs)
        else:
            raise TypeError
    for i in range(num_envs):
        json_data.append(
            {
                "env_id": base_env,
                **dict(map(lambda x: (x[0], x[1][i]), var_values.items())),
            }
        )
    return json_data


if __name__ == "__main__":
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

    assert case1 == generate_config_json("LavaGrid", 3, ["lava_on"], [bool])
    try:
        assert case2 == generate_config_json(
            "MiniGrid-SimpleCrossingS9N0-v0", 3, ["num_crossings"], [int], [(1, 4)]
        )
    except:
        print(
            generate_config_json(
                "MiniGrid-SimpleCrossingS9N0-v0", 3, ["num_crossings"], [int], [(1, 4)]
            )
        )
        print(case2)
