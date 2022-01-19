import gym_minigrid
import gym

from novgrid.novelty_generation.novelty_wrappers import NoveltyWrapper
from novgrid.wrappers import FlatObsWrapper


def make_env(env_name, log_dir, wrappers=[FlatObsWrapper]):
    '''
    I think that you have to have this function because the 
    vectorization code expects a function wrappers is a list
    '''
    def _init():
        env = gym.make(env_name)
        if wrappers:
            for wrapper in wrappers:
                env = wrapper(env)
        # env = Monitor(env, log_dir)
        # obs = env.reset()
        return env
    return _init

