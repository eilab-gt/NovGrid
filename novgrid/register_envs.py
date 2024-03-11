from gymnasium import Env
from gymnasium.envs.registration import register
import novgrid.envs as envs
import inspect


def register_novgrid_envs() -> None:
    [
        register(id=f"NovGrid-{name}", entry_point=f"novgrid.envs:{name}")
        for name, _ in inspect.getmembers(
            envs, lambda obj: inspect.isclass(obj) and issubclass(obj, Env)
        )
    ]
