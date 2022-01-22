import configargparse


def getparser():
    '''
    Reminder: all values have to be here to be modified by a config file.
    Precendence: command line > environment variables > config file values > defaults
    '''
    p = configargparse.ArgParser(default_config_files=['default.ini'])
    p.add('--exp_config', required=False, is_config_file=True, help='config file path for the experiment')
    p.add('--wandb_config', required=False, is_config_file=True, help='config file path for WandB')
    p.add('-t', '--total_timesteps', type=int, default=10000000, help='total timesteps per experiment')
    p.add('-e', '--env', type=str, default='MiniGrid-DoorKey-6x6-v0', help='Core environment')
    p.add('-s', '--saves_logs', type=str, default='minigrid_cnn_logs', help='where to save logs and models')
    p.add('-w', '--wandb_track', default=True, action='store_true', help='whether or not to set up as a wandb run')
    p.add('-n', '--novelty_step', type=int, help='step when novelty occurs. negative numbers turn off novelty')
    p.add('--load_model', type=str, default='', help='model to load. empty string learns from scratch') #models/best_model.zip')
    p.add('--num_exp', type=int, default=1, help='number of learning experiments per run')
    p.add('--learning_rate', type=float, default=1e-4, help='Learning rate for optimization')
    p.add('--num_workers', type=int, default=1, help='number of learning workers, and therefore environments')
    p.add('--seed', type=int, default=13, help='seed for randomness')
    p.add('--debug', default=False, action='store_true')
    p.add('--novelty_episode', default=10000, help='episode in which novelty is injected')
    parsed_args = p.parse_args()
    print(parsed_args)
    return parsed_args
