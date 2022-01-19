from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Door2KeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.place_obj(
            obj=Key('blue'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        
        self.mission = "use the key to open the door and then get to the goal"

class Door2KeyEnv6x6(Door2KeyEnv):
    def __init__(self):
        super().__init__(size=6)


register(
    id='MiniGrid-Door2Key-6x6-v0',
    entry_point='novgrid.envs:Door2KeyEnv6x6'
)
