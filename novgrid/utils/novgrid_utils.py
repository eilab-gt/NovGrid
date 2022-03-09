import gym

from novgrid.novelty_generation.novelty_wrappers import NoveltyWrapper


def make_env(env_name, log_dir, wrappers=[], novelty_episode=0):
    def _init():
        env = gym.make(env_name)
        if wrappers:
            for wrapper in wrappers:
                if issubclass(wrapper, NoveltyWrapper):
                    env = wrapper(env, novelty_episode=novelty_episode)
                else:
                    env = wrapper(env)
        return env
    return _init

