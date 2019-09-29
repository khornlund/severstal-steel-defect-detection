import yaml


def load_config(filename):
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    config['name'] = verbose_config_name(config)
    return config


def verbose_config_name(config):
    short_name = config['short_name']
    arch = f"{config['arch']['type']}-{config['arch']['args']['encoder_name']}"
    loss = config['loss']
    optim = config['optimizer']['type']
    return '-'.join([short_name, arch, loss, optim])
