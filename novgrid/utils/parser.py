import configargparse


def getparser():
    p = configargparse.ArgParser(default_config_files=['default.ini'])  #['/etc/app/conf.d/*.conf', '~/.my_settings'])
    p.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    # p.add('--genome', required=True, help='path to genome file')  # this option can be set in a config file because it starts with '--'
    p.add('-wandb', default='True',  action='store_true', help='whether or not to set up as a wandb run')
    p.add('-n', '--novelty-step', type=int, help='step when novelty occurs. negative numbers turn off novelty')

    # p.add('-d', '--dbsnp', help='known variants .vcf', env_var='DBSNP_PATH')  # this option can be set in a config file because it starts with '--'
    # p.add('vcf', nargs='+', help='variant file(s)')
    return p
