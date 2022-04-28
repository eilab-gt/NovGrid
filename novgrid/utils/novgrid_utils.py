from gym_minigrid.wrappers import FlatObsWrapper
import gym


from novgrid.novelty_generation.novelty_wrappers import NoveltyWrapper


def make_env(env_name, wrapper_args=None, wrappers=[], novelty_episode=0):
    '''
    I think that you have to have this function because the 
    vectorization code expects a function wrappers is a list

    Parameters
    ----------
    env_name : str
        Name of the environment
    wrapper_args : list of dicts
        List of dictionaries, where each dictionary contains arguments for the wrapper at the same index
    wrappers : list of functions
        List of Wrapper functions that will be applied to the environment
    novelty_episode : int
        Episode number for novelty generation. To be deprecated
    '''
    def _init():
        env = gym.make(env_name)
        ## Check to make sure that there are the same number of arg dicts as wrappers
        if isinstance(wrapper_args, list):
            assert len(wrapper_args) == len(wrappers)
        if wrappers:
            for idx, wrapper in enumerate(wrappers):
                if isinstance(wrapper_args,list):
                    if issubclass(wrapper, NoveltyWrapper):
                        print("DEPRECATION WARNING: NoveltyWrapper should be redesigned with novelty_episode as a wrapper arg")
                        env = wrapper(env, novelty_episode=novelty_episode, *wrapper_args)
                    else:
                        env = wrapper(env, **wrapper_args[idx])
                else:
                    if issubclass(wrapper, NoveltyWrapper):
                        env = wrapper(env, novelty_episode=novelty_episode)
                    else:
                        env = wrapper(env)
        # env = FlatObsWrapper(env)
        # env = Monitor(env, wrapper_args)
        # obs = env.reset()
        return env
    return _init

