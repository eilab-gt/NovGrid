# change the self.mission
# you should be able to specify the exact novelty AND that there should be a random novelty
import abc
import gym
import numpy as np

from .novelty_objs import ColorDoor, MultiKeyDoor
from gym_minigrid.minigrid import Key, Grid, Door, Goal


class NoveltyWrapper(gym.core.Wrapper):
    """
    Wrapper to modify the environment according to novelty ontology at a certain point
    If novelty_episode = 0 (default) then there is no novelty
    This make the assumption--which is valid for all standard Minigrid environments as of 2021
    that MiniGrid environments do not ever overload the `reset()` gym function. So, we will assume
    that all relevant novelties are to be implemented in the `_post_novelty_gen_grid` function,
    which is called by `_post_novelty_reset` after the novelty episode is reached.
    """

    def __init__(self, env, novelty_episode):
        super().__init__(env)
        self.novelty_episode = novelty_episode
        self.num_episodes = 0

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

    # @abc.abstractmethod
    def _post_novelty_gen_grid(self, width, height):
        """
        This is the main function where you implement the novelty
        """
        raise NotImplementedError

    def _rand_int(self, low, high):
        return self.env.np_random.randint(low, high)


class DoorKeyChange(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

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
        doorIdx = self._rand_int(1, width-2)
        # Yellow door object that will open when toggled with a blue key
        self.env.put_obj(ColorDoor('yellow', is_locked=True, key_color='blue'), splitIdx, doorIdx)

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


class DoorLockToggle(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

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
        # Yellow door object that is already unlocked
        self.env.put_obj(Door('yellow', is_locked=False), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.env.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.env.mission = "go through the unlocked door and then get to the goal"


class DoorNumKeys(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        # Create an empty grid
        self.env.grid = Grid(width, height)

        # Generate the surrounding walls
        self.env.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.env.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(3, width - 2)
        self.env.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.env.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        # Yellow door that requires a yellow key and a blue key to be opened
        self.env.put_obj(MultiKeyDoor(
            'yellow', 
            is_locked=True, 
            key_colors=['yellow', 'blue']), 
        splitIdx, doorIdx)

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

        self.env.mission = "use two keys to open the door and then get to the goal"


class GoalLocationChange(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

    def _post_novelty_gen_grid(self, width, height):
        # Create an empty grid
        self.env.grid = Grid(width, height)

        # Generate the surrounding walls
        self.env.grid.wall_rect(0, 0, width, height)

        # Changes the location of the goal from the bottom-right corner to the top-right corner
        self.env.put_obj(Goal(), width - 2, 1)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.env.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.env.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.env.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.env.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.env.mission = "use the key to open the door and then get to the goal whose location has changed"


class ImperviousToLava(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

    def reset(self, **kwargs):
        self.num_episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        if self.num_episodes >= self.novelty_episode:
            fwd_pos = self.env.front_pos
            fwd_cell = self.env.grid.get(*fwd_pos)
            obs, reward, done, info = self.env.step(action, **kwargs)
            if done and fwd_cell and fwd_cell.type == 'lava':
                self.env.agent_pos = fwd_pos
                obs = self.env.gen_obs()['image']
                done = False
            return obs, reward, done, info
        return self.env.step(action, **kwargs)


class ForwardMovementSpeed(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

    def reset(self, **kwargs):
        self.num_episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        if self.num_episodes >= self.novelty_episode:
            if action == self.env.actions.forward:
                obs, reward, done, info = self.env.step(action, **kwargs)
                if done:
                    return obs, reward, done, info
                self.env.step_count -= 1
        return self.env.step(action, **kwargs)


class ActionReptition(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)
        self.prev_action = None

    def reset(self, **kwargs):
        self.num_episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        if self.num_episodes >= self.novelty_episode:
            if action != self.prev_action:
                self.prev_action = action
                return self.env.step(self.env.actions.done)
            self.prev_action = None
        return self.env.step(action, **kwargs)
    
    
class ActionRadius(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

    def reset(self, **kwargs):
        self.num_episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        if self.num_episodes >= self.novelty_episode:
            obs, reward, done, info = self.env.step(action, **kwargs)
            if action == self.env.actions.pickup and self.env.carrying is None:
                agent_pos = self.env.agent_pos
                self.env.step(self.env.actions.forward, **kwargs)
                self.env.step(action, **kwargs)
                self.env.agent_pos = agent_pos
                self.env.step_count -= 2
                obs = self.env.gen_obs()
            return obs, reward, done, info
        return self.env.step(action, **kwargs)


class Burdening(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

    def reset(self, **kwargs):
        self.num_episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        if self.num_episodes >= self.novelty_episode:
            if action == self.env.actions.forward and self.env.carrying:
                self.env.step_count += 1
            elif action == self.env.actions.forward and not self.env.carrying:
                obs, reward, done, info = self.env.step(action, **kwargs)
                if done:
                    return obs, reward, done, info
                self.env.step_count -= 1
        return self.env.step(action, **kwargs)


class ColorRestriction(NoveltyWrapper):

    def __init__(self, env, novelty_episode):
        super().__init__(env, novelty_episode)

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
        self.env.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        doorIdx = self._rand_int(1, width - 2)
        while isinstance(self.env.grid.get(splitIdx, doorIdx), Door):
            doorIdx = self._rand_int(1, width - 2)
        self.env.put_obj(Door('blue', is_locked=True), splitIdx, doorIdx)

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

        self.env.mission = "use blue key to open the blue door and then get to the goal"


    def step(self, action, **kwargs):
        if self.num_episodes >= self.novelty_episode:
            if action == self.env.actions.pickup:
                fwd_pos = self.env.front_pos
                fwd_cell = self.env.grid.get(*fwd_pos)
                if fwd_cell and fwd_cell.can_pickup() and fwd_cell.color == 'yellow':
                    return self.env.step(self.env.actions.done, **kwargs)
        return self.env.step(action, **kwargs)





    
