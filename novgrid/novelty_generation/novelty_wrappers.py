# change the self.mission
# you should be able to specify the exact novelty AND that there should be a random novelty
import abc
import gym
import numpy as np

from .novelty_objs import ColorDoor
from gym_minigrid.minigrid import Key, Grid, Door, Goal, COLORS


class NoveltyWrapper(gym.core.Wrapper):
    """
    Wrapper to modify the environment according to novelty ontology at a certain point
    If novelty_episode = 0 (default) then there is no novelty
    This make the assumption--which is valid for all standard Minigrid environments as of 2021
    that MiniGrid environments do not ever overload the `reset()` gym function. So, we will assume
    that all relevant novelties are to be implemented in the `_post_novelty_gen_grid` function,
    which is called by `_post_novelty_reset` after the novelty episode is reached.
    """

    def __init__(self, env, novelty_episode=0):
        super().__init__(env)
        self.novelty_episode = novelty_episode
        self.novelty_injected = False

    def reset(self, **kwargs):
        self.num_episodes += 1
        if self.num_episodes >= self.novelty_episode:
            return self._post_novelty_reset(**kwargs)
        else:
            return self.env.reset(**kwargs)

    def _post_novelty_reset(self, **kwargs):
        # Current position and direction of the agent
        self.env.agent_pos = None
        self.env.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._post_novelty_gen_grid(self.width, self.height, **kwargs)

        # These fields should be defined by _gen_grid
        assert self.env.agent_pos is not None
        assert self.env.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.env.grid.get(*self.env.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.env.carrying = None

        # Step count since episode start
        self.env.step_count = 0

        # Return first observation
        obs = self.env.gen_obs()
        return obs

    def _rand_int(self, low, high):
        return self.env.np_random.randint(low, high)

    # @abc.abstractmethod
    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class DoorKeyNoveltyWrapper(NoveltyWrapper):

    def _post_novelty_gen_grid(self, width, height):
        # Create an empty grid
        self.env.grid = Grid(width, height)

        # Generate the surrounding walls
        self.env.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.env.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.env.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.env.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.env.put_obj(
            ColorDoor('yellow', is_locked=True, key_color='blue'),
            splitIdx,
            doorIdx
        )

        # Place a blue key on the left side
        self.env.place_obj(
            obj=Key('blue'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        self.env.mission = "use different color key to open the door and then get to the goal"


class Door2KeyNoveltyWrapper(NoveltyWrapper):

    def _post_novelty_gen_grid(self, width, height):
        # Create an empty grid
        self.env.grid = Grid(width, height)

        # Generate the surrounding walls
        self.env.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.env.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.env.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.env.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.env.put_obj(
            ColorDoor('yellow', is_locked=True, key_color='blue'),
            splitIdx,
            doorIdx
        )

        # Place a yellow key on the left side
        self.env.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        # Place a blue key on the left side
        self.env.place_obj(
            obj=Key('blue'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        self.env.mission = "use different color key to open the door and then get to the goal"


class MultiDoorMultiKeyNoveltyWrapper(NoveltyWrapper):

    def _post_novelty_gen_grid(self, width, height):
        # Create an empty grid
        self.env.grid = Grid(width, height)

        # Generate the surrounding walls
        self.env.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.env.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        if self.env.determ:
            split_idx = self.env.split_idx
        else:
            split_idx = self._rand_int(2, width - 2)
        self.env.grid.vert_wall(split_idx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.env.place_agent(size=(split_idx, height))

        # Place doors and keys
        # Warning: for Python < 3.5 dict order is non-deterministic
        colors = list(COLORS.keys())
        rand_num_gen = np.random.default_rng(self.env.seed_value)
        # place_obj drops the object randomly in a rectangle
        # put_obj puts an object in a specific place
        for door in range(self.env.doors):
            if self.env.determ:
                door_idx = self.env.door_idxs[door]
            else:
                door_idx = rand_num_gen.choice(height - 3) + 1
            key_color = (door + 1) % max(2, self.env.doors, self.env.keys)
            self.env.put_obj(
                ColorDoor(colors[door], is_locked=True, key_color=key_color),
                split_idx,
                door_idx
            )

        for key in range(self.env.keys):
            if self.env.determ:
                self.env.put_obj(Key(colors[key]), self.env.key_widths[key], self.env.key_heights[key])
            else:
                self.env.place_obj(obj=Key(colors[key]), top=(0, 0), size=(split_idx, height))

        self.env.mission = "use different color keys to open doors and then get to the goal"
