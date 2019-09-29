import click

import torch

from sever.main import Master
from sever.utils import kaggle_upload, load_config


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
        Master.start(config, resume)
