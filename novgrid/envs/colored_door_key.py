from typing import Optional, Any, Dict, List

from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Key
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv

from novgrid.envs.novgrid_objects import ColorDoor


class ColoredDoorKeyEnv(MiniGridEnv):

    def __init__(
        self,
        door_color: str = "yellow",
        key_colors: Optional[List[str]] = None,
        correct_key_color: str = "yellow",
        size: int = 8,
        max_steps: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ):
        self.door_color = door_color
        self.key_colors = key_colors if key_colors is not None else [correct_key_color]
        self.correct_key_color = correct_key_color
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the correct key to open the door and get to the goal"

    def step(self, action):
        return super().step(action)

    def _gen_grid(self, width: int, height: int):
        # Create an empty grid
        self.grid = Grid(width=width, height=height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation on the left side
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(
            ColorDoor(
                self.door_color, is_locked=True, key_color=self.correct_key_color
            ),
            splitIdx,
            doorIdx,
        )

        # Place a yellow key on the left side
        for color in self.key_colors:
            self.place_obj(obj=Key(color=color), top=(0, 0), size=(splitIdx, height))

        self.mission = self._gen_mission()
