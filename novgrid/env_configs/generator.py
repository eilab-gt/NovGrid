import abc
import os

import json
from typing import List, Any, Dict
import numpy as np


class Change(abc.ABC):
    """
    Abstract class representing a change to be applied during environment configuration generation.
    """

    def __init__(self) -> None:
        """Initialization method for the Change class."""
        pass

    @abc.abstractmethod
    def generate_value(self, i: int, num_tasks: int) -> Any:
        """
        Abstract method to generate a value based on the change.

        Args:
            i (int): Current iteration.
            num_tasks (int): Total number of tasks.

        Returns:
            Any: Generated value.
        """
        pass


class Constant(Change):
    """
    A constant value change to be applied during environment configuration generation.
    """

    def __init__(self, x: Any) -> None:
        """
        Initializes the Constant change with a fixed value.

        Args:
            x (Any): Constant value.
        """
        self.x = x

    def generate_value(self, i: int, num_tasks: int) -> Any:
        """
        Generates the constant value.

        Args:
            i (int): Current iteration.
            num_tasks (int): Total number of tasks.

        Returns:
            Any: Constant value.
        """
        return self.x


class IntRange(Change):
    """
    An integer range change to be applied during environment configuration generation.
    """

    def __init__(self, start: int, end: int, inclusive: bool = False) -> None:
        """
        Initializes the IntRange change with a specified range.

        Args:
            start (int): Start of the range.
            end (int): End of the range.
            inclusive (bool): Whether to include the end value in the range.
        """
        super().__init__()
        self.start = start
        self.end = end
        self.inclusive = inclusive

    def generate_value(self, i: int, num_tasks: int) -> int:
        """
        Generates an integer value within the specified range.

        Args:
            i (int): Current iteration.
            num_tasks (int): Total number of tasks.

        Returns:
            int: Generated integer value.
        """
        return (
            self.start + i * (self.end + int(self.inclusive) - self.start) // num_tasks
        )


class FloatRange(Change):
    """
    A float range change to be applied during environment configuration generation.
    """

    def __init__(self, start: float, end: float, inclusive: bool = False) -> None:
        """
        Initializes the FloatRange change with a specified range.

        Args:
            start (float): Start of the range.
            end (float): End of the range.
            inclusive (bool): Whether to include the end value in the range.
        """
        super().__init__()
        self.start = start
        self.end = end
        self.inclusive = inclusive
        self._linspace_cache = {}

    def generate_value(self, i: int, num_tasks: int) -> float:
        """
        Generates a float value within the specified range.

        Args:
            i (int): Current iteration.
            num_tasks (int): Total number of tasks.

        Returns:
            float: Generated float value.
        """
        if num_tasks not in self._linspace_cache:
            self._linspace_cache[num_tasks] = np.linspace(
                self.start, self.end, num=num_tasks, endpoint=self.inclusive
            )
        return self._linspace_cache[num_tasks][i]


class Toggle(Change):
    """
    A toggle change to be applied during environment configuration generation.
    """

    def __init__(self, val1: Any = False, val2: Any = True) -> None:
        """
        Initializes the Toggle change with two values.

        Args:
            val1 (Any): First value.
            val2 (Any): Second value.
        """
        super().__init__()
        self.val1 = val1
        self.val2 = val2

    def generate_value(self, i: int, num_tasks: int) -> Any:
        """
        Generates one of the two values based on the iteration index.

        Args:
            i (int): Current iteration.
            num_tasks (int): Total number of tasks.

        Returns:
            Any: Generated value.
        """
        return self.val1 if i % 2 == 0 else self.val2


class ListChange(Change):
    """
    A list change to be applied during environment configuration generation.
    """

    def __init__(self, lst: List[Any], use_snake_boundary: bool = False) -> None:
        """
        Initializes the ListChange with a list of values.

        Args:
            lst (List[Any]): List of values.
            use_snake_boundary (bool): Whether to use a snake-like boundary for the list.
        """
        super().__init__()
        self.lst = lst
        if use_snake_boundary:
            self.lst = self.lst + self.lst[-2:0:-1]

    def generate_value(self, i: int, num_tasks: int) -> Any:
        """
        Generates a value from the list based on the iteration index.

        Args:
            i (int): Current iteration.
            num_tasks (int): Total number of tasks.

        Returns:
            Any: Generated value.
        """
        return self.lst[i % len(self.lst)]


class EnvConfigGenerator:
    """
    Class for generating environment configurations based on specified changes.
    """

    def __init__(self, env_id: str, num_tasks: int, changes: Dict[str, Change]) -> None:
        """
        Initializes the EnvConfigGenerator with the base environment ID, number of tasks, and changes.

        Args:
            env_id (str): Base environment ID.
            num_tasks (int): Number of tasks to generate.
            changes (Dict[str, Change]): Dictionary of changes to be applied.
        """
        self.base_env_id = env_id
        self.num_tasks = num_tasks
        self.changes = changes

    def generate_env_configs(self) -> List[Dict[str, Any]]:
        """
        Generates environment configurations based on the specified changes.

        Returns:
            List[Dict[str, Any]]: List of environment configurations.
        """
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
        """
        Generates and saves environment configurations to a JSON file.

        Args:
            json_file_name (str): Name of the JSON file.

        Returns:
            List[Dict[str, Any]]: List of environment configurations.
        """
        env_configs = self.generate_env_configs()
        with open(json_file_name, "w") as f:
            json.dump(env_configs, f)
        return env_configs

    def global_save_env_configs(
        self, name: str, override_existing_file: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generates and globally saves environment configurations.

        Args:
            name (str): Name of the environment configuration.
            override_existing_file (bool): Whether to override an existing configuration file.

        Returns:
            List[Dict[str, Any]]: List of environment configurations.
        """
        fname = f"{name}.json" if ".json" not in name else name
        full_fname = os.path.join(os.path.dirname(__file__), "json", fname)
        if os.path.exists(full_fname) and not override_existing_file:
            raise ValueError(
                f"The name {name} already has a global file for its env config. To override this file use the override_existing_file flag."
            )
        self.save_env_configs(full_fname)


def test_generator_bool_toggle():
    """
    Test case for EnvConfigGenerator with a boolean toggle change.
    """
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
    """
    Test case for EnvConfigGenerator with an integer range change.
    """
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
    """
    Test case for EnvConfigGenerator with a list change.
    """
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
        changes={"num_crossings": ListChange([1, 2, 3], use_snake_boundary=True)},
    ).generate_env_configs()

    assert expected_result == result


def test_generator_float_range():
    """
    Test case for EnvConfigGenerator with a float range change.
    """
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
    """
    Test case for EnvConfigGenerator with multiple changes.
    """
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
