import gymnasium as gym

from novgrid.novelty_generation.novelty_wrappers import NoveltyWrapper


def make_env(env_name, wrappers=None, wrapper_args=None, novelty_episode=-1):
    '''
    I think that you have to have this function because the 
    vectorization code expects a function wrappers is a list

    Parameters
    ----------
    env_name : str
        Name of the environment
    wrappers : list of functions
        List of Wrapper functions that will be applied to the environment
    wrapper_args : list of dicts
        List of dictionaries, where each dictionary contains arguments for the wrapper at the same index
    novelty_episode : int
        Episode number for novelty generation. To be deprecated
    '''
    if wrappers is None:
        wrappers = []

    def _init():
        env = gym.make(env_name)
        ## Check to make sure that there are the same number of arg dicts as wrappers
        if wrapper_args is not None:
            assert len(wrapper_args) == len(wrappers)
        if wrappers:
            for idx, wrapper in enumerate(wrappers):
                if wrapper_args[idx] is not None:
                    if issubclass(wrapper, NoveltyWrapper) and novelty_episode > 0:
                        print("DEPRECATION WARNING: NoveltyWrapper should be redesigned with novelty_episode as a wrapper arg")
                        env = wrapper(env, novelty_episode=novelty_episode, **wrapper_args[idx])
                    else:
                        env = wrapper(env, **wrapper_args[idx])
                else:
                    if issubclass(wrapper, NoveltyWrapper) and novelty_episode > 0:
                        print("DEPRECATION WARNING: NoveltyWrapper should be redesigned with novelty_episode as a wrapper arg")
                        env = wrapper(env, novelty_episode=novelty_episode)
                    else:
                        env = wrapper(env)
        return env
    return _init

