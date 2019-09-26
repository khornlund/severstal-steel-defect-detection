import click
import os
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
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def train(config_filename, resume, device):
    if config_filename:
        configs = [load_config(f) for f in config_filename]
    elif resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with
        # changed configurations.
        configs = [torch.load(resume)['config']]
    else:
        raise AssertionError('Configuration file need to be specified. '
                             'Add "-c experiments/config.yaml", for example.')

    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    for config in configs:
        Runner().train(config, resume)


@cli.command()
@click.option('-c', '--config-filename', default='experiments/config.yml', type=str,
              help='config file path (default: None)')
@click.option('-m', '--model-checkpoint', default=None, type=str,
              help='path to latest checkpoint (default: None)')
@click.option('-d', '--device', default=None, type=str,
              help='indices of GPUs to enable (default: all)')
def predict(config_filename, model_checkpoint, device):
    config = load_config(config_filename)
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    Runner().predict(config, model_checkpoint)


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
