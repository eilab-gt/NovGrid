from typing import Optional, Any, Dict

from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv


class LavaShortcutMaze(MiniGridEnv):
    """
    Environment with a door and key with one wall of lava with a small gap to cross through
    sparse reward
    """

    def __init__(
        self,
        obstacle_type: WorldObj = Lava,
        lava_safe: bool = False,
        size: int = 8,
        max_steps: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ):
        self.obstacle_type = obstacle_type
        self.fixed_env = True
        self.simple_reward = False
        self.lava_safe = lava_safe

        if max_steps is None:
            max_steps = 10 * size**2

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return (
            "Avoid the lava and use the key to open the door and then get to the goal"
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # first vertical walls
        first_wall_width = 2
        splitIdx = width // 2
        self.grid.vert_wall(first_wall_width, 2, height - 3)

        if width > 6:
            for extra_wall_pos in range(1, (width - 5) // 2 + 1):
                if extra_wall_pos % 2 == 0:
                    self.grid.vert_wall(
                        first_wall_width + extra_wall_pos * 2, 2, height - 3
                    )
                    # Place a goal in the bottom-right corner
                    # self.put_obj(Goal(), width - 2, height - 2)
                else:
                    self.grid.vert_wall(
                        first_wall_width + extra_wall_pos * 2, 0, height - 3
                    )
                    # Place a goal in the top-right corner
                    # self.put_obj(Goal(), width - 2, 1)
        else:
            pass
        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a horizontal lava
        self.grid.horz_wall(2, height - 2, width - 4, Lava)

        # Place the agent at a fixed bottom left position
        # and random orientation
        self.place_agent(top=(0, height - 2), size=(first_wall_width, height))

        self.mission = (
            "Avoid the lava and use the key to open the door and then get to the goal"
        )

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        agent_pos = self.agent_pos
        object = self.grid.get(agent_pos[0], agent_pos[1])
        if object.type == "lava":
            return -1  # Add Negative reward for stepping on Lava.

        if self.simple_reward:
            return 1
        else:
            return 1 - 0.9 * (self.step_count / self.max_steps)  # * 10

    def step(self, action):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)
        if (
            self.lava_safe
            and terminated
            and action == self.actions.forward
            and fwd_cell
            and fwd_cell.type == "lava"
        ):
            self.agent_pos = fwd_pos
            obs = self.gen_obs()
            terminated = False
        return obs, reward, terminated, truncated, info
