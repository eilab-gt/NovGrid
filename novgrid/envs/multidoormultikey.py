from gym_minigrid.envs.doorkey import DoorKeyEnv
from gym_minigrid.minigrid import Key, Grid, Door, Goal, COLORS
from gym_minigrid.register import register
import numpy as np

class MultiDoorMultiKeyEnv(DoorKeyEnv):
    def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
        if (doors > (size-3)) or (keys > (size-3)):
            raise ValueError("Both doors:{} and keys:{} must be less than size-3:{}".format(doors, keys, size))
        elif doors > 6 or keys > 6:
            raise ValueError("Both doors:{} and keys:{} must be less than 6".format(doors, keys))
        self.doors = doors
        self.keys = keys
        self.seed = seed
        self.determ = determ
        if self.determ:
            rand_num_gen = np.random.default_rng(self.seed)
            self.door_idxs = rand_num_gen.choice(size-3, size=self.doors)+1
            self.key_widths = rand_num_gen.choice(size, size=self.keys)
            self.key_heights = rand_num_gen.choice(size, size=self.keys)
            self.split_idx = rand_num_gen.integers(low=2, high=size-2)
        super().__init__(size=16)

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
            split_idx = self._rand_int(2, width-2)
        self.grid.vert_wall(split_idx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(split_idx, height))

        ## Place doors and keys
        ## Warning: for Python < 3.5 dict order is non-deterministic
        colors = COLORS.keys()
        rand_num_gen = np.random.default_rng(self.seed)
        # place_obj drops the object randomly in a rectangle
        # put_obj puts an object in a specific place
        for door in range(self.doors):
            if self.determ:
                door_idx = self.door_idxs[door]
            else:
                door_idx = rand_num_gen.choice(height-3)+1
            self.put_obj(Door(colors[door], is_locked=True), split_idx, door_idx)

        for key in range(self.keys):
            if self.determ:
                self.put_obj(Key(colors[key]), self.key_widths[key], self.key_heights[key])

            self.place_obj(obj=Key(colors[key]), top=(0, 0), size=(split_idx, height))

        self.mission = "use the key to open the same color door and then get to the goal"


class DoorMultiKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)


class DoorMultiKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)


class DoorMultiKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)


register(
    id='MiniGrid-DoorMultiKey-5x5-v0',
    entry_point='minigrid_novelty_generator.envs:DoorMultiKeyEnvEnv5x5'
)

register(
    id='MiniGrid-DoorMultiKey-6x6-v0',
    entry_point='minigrid_novelty_generator.envs:DoorMultiKeyEnv6x6'
)

register(
    id='MiniGrid-DoorMultiKey-8x8-v0',
    entry_point='minigrid_novelty_generator.envs:DoorMultiKeyEnv'
)

register(
    id='MiniGrid-DoorMultiKey-16x16-v0',
    entry_point='minigrid_novelty_generator.envs:DoorMultiKeyEnv16x16'
)
