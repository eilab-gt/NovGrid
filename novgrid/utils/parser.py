import configargparse


def getparser(inputs=None):
    """
    Reminder: all values have to be here to be modified by a config file.
    Precendence: command line > environment variables > config file values > defaults
    """
    p = configargparse.ArgParser(default_config_files=['default.ini'])
    p.add('--exp_config', required=False, is_config_file=True, help='config file path for the experiment')
    p.add('-t', '--total_timesteps', type=int, default=2500000, help='total timesteps per experiment')
    p.add('-e', '--env', type=str, default='MiniGrid-DoorKey-8x8-v0', help='Core environment')
    p.add('-s', '--saves_logs', type=str, default='novgrid_logs', help='where to save logs and models')
    p.add('--device', type=str, default='', help='device. code assumes empty means to autocheck')
    p.add('--load_model', type=str, default='', help='model to load. empty string learns from scratch') #models/best_model.zip')
    p.add('--num_exp', type=int, default=1, help='number of learning experiments per run')
    p.add('-w', '--wandb_track', default=False, action='store_true', help='whether or not to set up as a wandb run')
    p.add('--learning_rate', type=float, default=2.5e-4, help='Learning rate for optimization')
    p.add('--num_workers', type=int, default=1, help='number of learning workers, and therefore environments')
    p.add('--seed', type=int, default=13, help='seed for randomness')
    p.add('--debug', default=False, action='store_true')
    p.add('--novelty_wrapper', type=str, default='', help='novelty to inject into environment')
    p.add('--novelty_episode', type=int, default=10000, help='episode in which novelty is injected')
    p.add('--eval_interval', type=int, default=1000, help='how many steps between evaluatations')
    p.add('--log_interval', type=int, default=10, help='how many steps between logging')
    
    if inputs is None:
        parsed_args = p.parse_args()
    else:
        parsed_args = p.parse_args(inputs)
    print(parsed_args)
    return parsed_args
