from minigrid.envs import DoorKeyEnv
from gymnasium.envs.registration import register
from minigrid.core.world_object import Key, Door, Goal
from minigrid.core.grid import Grid
from minigrid.core.constants import COLORS
from matplotlib.pyplot import grid
import numpy as np


class MultiDoorMultiKeyEnv(DoorKeyEnv):
    def __init__(self, size=6, doors=1, keys=1, determ=False, seed=13, locked=None):
        if (doors > (size - 3)) or (keys > (size - 3)):
            raise ValueError("Both doors:{} and keys:{} must be less than size-3:{}".format(doors, keys, size))
        elif doors > 6 or keys > 6:
            raise ValueError("Both doors:{} and keys:{} must be less than 6".format(doors, keys))
        self.doors = doors
        self.keys = keys
        self.seed_value = seed
        self.determ = determ
        if self.determ:
            rand_num_gen = np.random.default_rng(self.seed_value)
            self.door_idxs = rand_num_gen.choice(size - 3, size=self.doors, replace=False) + 1
            self.key_widths = rand_num_gen.choice(size, size=self.keys)
            self.key_heights = rand_num_gen.choice(size, size=self.keys)
            self.split_idx = rand_num_gen.integers(low=2, high=size - 2)
        super().__init__(size=size)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        if self.determ:
            split_idx = self.split_idx
        else:
            split_idx = self._rand_int(2, width - 2)
        self.grid.vert_wall(split_idx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(split_idx, height))

        ## Place doors and keys
        ## Warning: for Python < 3.5 dict order is non-deterministic
        colors = list(COLORS.keys())
        rand_num_gen = np.random.default_rng(self.seed_value)
        # place_obj drops the object randomly in a rectangle
        # put_obj puts an object in a specific place
        for door in range(self.doors):
            if self.determ:
                door_idx = self.door_idxs[door]
            else:
                door_idx = None
                while not door_idx or isinstance(self.grid.get(split_idx, door_idx), Door):
                  door_idx = rand_num_gen.choice(height - 3) + 1
            self.put_obj(Door(colors[door], is_locked=True), split_idx, door_idx)

        for key in range(self.keys):
            if self.determ:
                self.put_obj(Key(colors[key]), self.key_widths[key], self.key_heights[key])
            self.place_obj(obj=Key(colors[key]), top=(0, 0), size=(split_idx, height))

        self.mission = "use the key to open the same color door and then get to the goal"

class DoorMultiKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5, doors=2, keys=2)

class DoorMultiKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6, doors=2, keys=2)

class DoorMultiKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, doors=2, keys=2)

register(
    id='MiniGrid-DoorMultiKey-5x5-v0',
    entry_point='novgrid.envs:DoorMultiKeyEnvEnv5x5'
)

register(
    id='MiniGrid-DoorMultiKey-6x6-v0',
    entry_point='novgrid.envs:DoorMultiKeyEnv6x6'
)

register(
    id='MiniGrid-DoorMultiKey-16x16-v0',
    entry_point='novgrid.envs:DoorMultiKeyEnv16x16'
)
