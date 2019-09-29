import click
import yaml

import torch

from sever.main import Runner
from sever.utils import kaggle_upload


@click.group()
def cli():
    """CLI for sever"""


@cli.command()
@click.option('-r', '--run-directory', required=True, type=str, help='Path to run')
@click.option('-e', '--epochs', type=int, multiple=True, help='Epochs to upload')
def upload(run_directory, epochs):
    """Upload model weights as a dataset to kaggle"""
    kaggle_upload(run_directory, epochs)


@cli.command()
@click.option('-c', '--config-filename', type=str, multiple=True,
              help='config file path (default: None)')
@click.option('-r', '--resume', default=None, type=str,
              help='path to latest checkpoint (default: None)')
def train(config_filename, resume):
    if config_filename:
        configs = [load_config(f) for f in config_filename]
    elif resume:
        configs = [torch.load(resume)['config']]
    else:
        raise AssertionError('Configuration file need to be specified. '
                             'Add "-c experiments/config.yaml", for example.')
    for config in configs:
        Runner(config).train(resume)


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
