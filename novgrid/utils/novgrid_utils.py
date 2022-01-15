import gym_minigrid
import gym

from novgrid.novelty_generation.novelty_wrappers import NoveltyWrapper


def make_env(env_name, log_dir, wrappers=[gym_minigrid.wrappers.ImgObsWrapper]):
    '''
    I think that you have to have this function because the 
    vectorization code expects a function wrappers is a list
    '''
    def _init():
        env = gym.make(env_name)
        if wrappers:
            for wrapper in wrappers:
                if issubclass(wrapper, NoveltyWrapper):
                    env = wrapper(env, novelty_episode=10_000)
                else:
                    env = wrapper(env)
        env = gym_minigrid.wrappers.ImgObsWrapper(env)
        # env = Monitor(env, log_dir)
        # obs = env.reset()
        return env
    return _init

