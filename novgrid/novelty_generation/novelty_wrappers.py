# change the self.mission
# you should be able to specify the exact novelty AND that there should be a random novelty
import abc
import gym


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
        This is the main function where you implement he novelty
        """
        raise NotImplementedError

