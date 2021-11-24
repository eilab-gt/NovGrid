# change the self.mission
# you should be able to specify the exact novelty AND that there should be a random novelty
import abc
import gym
import numpy as np
from gym_minigrid.envs import DoorKeyEnv
from novgrid.envs import *
from novgrid.novelty_generation.novelty_objs import ColorDoor


class NoveltyWrapper(gym.core.Wrapper):
    """
    Wrapper to modify the environment according to novelty ontology at a certain point
    If novelty_episode = 0 (default) then there is no novelty

    This make the assumption--which is valid for all standard Minigrid environments as of 2021--
    that MiniGrid environments do not ever overload the `reset()` gym function. So, we will assume
    that all relevant novelties are to be implemented in the `_post_novelty_gen_grid` function,
    which is called by `_post_novelty_reset` after the novelty episode is reached.
    """
    def __init__(self, env, novelty_episode=0):
        super().__init__(env)
        self.orig_env = env
        self.novelty_episode = novelty_episode
        self.num_episodes = 0

    def reset(self, **kwargs):
        self.num_episodes += 0
        if self.num_episodes >= self.novelty_episode:
            return self._post_novelty_reset(**kwargs)
        else:
            return self.env.reset(**kwargs)

    def _post_novelty_reset(self, **kwargs):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._post_novelty_gen_grid(self.width, self.height, **kwargs)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    # @abc.abstractmethod
    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class DoorKeyChange(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)
        # self.rand_num_gen = np.random.default_rng()

    def _post_novelty_gen_grid(self, width, height):
        """
        TODO: is there a way to implement this based around a call to
        super()._gen_grid(self, width, height)
        """
        from gym_minigrid.minigrid import Key, Grid, Door, Goal, COLORS
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        # TODO balloch: make the line below less hacky
        split_idx = self.env.env._rand_int(2, width-2)
        # split_idx = self.split_idx
        self.grid.vert_wall(split_idx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(split_idx, height))

        ## Place doors and keys
        ## Warning: for Python < 3.5 dict order is non-deterministic
        colors = COLORS.keys()
        # place_obj drops the object randomly in a rectangle
        # put_obj puts an object in a specific place
        # TODO balloch: this should reflect the number of keys in the original env
        door_idx = self.env.env._rand_int(1, width-2)
        self.put_obj(ColorDoor('yellow', is_locked=True, key_color='blue'), split_idx, door_idx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(split_idx, height)
        )

        # Place a blue key on the left side
        self.place_obj(
            obj=Key('blue'),
            top=(0, 0),
            size=(split_idx, height)
        )
        self.mission = "use the key to open the same color door and then get to the goal"


class DoorLockToggle(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class GoalTopologyChange(NoveltyWrapper):
    """

    """
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class NumKeys(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class NumDoors(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class ImperviousToLava(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class ActionRepetition(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class ColorRestriction(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class ActionRadius(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class TransitionDeterminism(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class Burdening(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


class ForwardMoveSpeed(NoveltyWrapper):
    # def __init__(self, size=8, doors=2, keys=2, determ=False, seed=13):
    def __init__(self, env, novelty_episode=0):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError


