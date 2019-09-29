import os
import random

from apex import amp
# from apex.parallel import convert_syncbn_model  # noqa
import numpy as np
import torch
import segmentation_models_pytorch as module_arch
import horovod.torch as hvd
import sever.data_loader.data_loaders as module_data
import sever.model.loss as module_loss
import sever.model.metric as module_metric
import sever.model.optimizer as module_optimizer
import sever.data_loader.augmentation as module_aug
from sever.trainer import Trainer
from sever.utils import setup_logger, setup_logging, load_config


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Worker:

    def __init__(self, config):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])
        self.config = config

    def train(self, resume=''):
        config = self.config
        rank       = hvd.rank()
        local_rank = hvd.local_rank()
        world_size = hvd.size()
        self.logger.info(f'Worker {local_rank+1}/{world_size} starting...')

        config['rank']       = rank
        config['local_rank'] = local_rank
        config['world_size'] = world_size

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        device = torch.device(f'cuda:{local_rank}')
        self.logger.info(f'Using device {device}')
        torch.cuda.set_device(device)
        model = model.to(device)

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(module_optimizer, 'optimizer', config, trainable_params)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

        opt_level = config['apex']
        self.logger.debug(f'Setting apex opt_level: {opt_level}')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        self.logger.info(f'Using `DistributedDataParallel`')
        model, optimizer = self._resume_checkpoint(resume, rank, model, optimizer)
        peek_weights = model.encoder._conv_stem.weight[0]
        self.logger.debug(f'Peek weights: {peek_weights}')

        self.logger.debug('Getting augmentations')
        transforms = getattr(module_aug, config['augmentation'])()

        self.logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config, rank, world_size, transforms)
        valid_data_loader = data_loader.split_validation()

        self.logger.debug('Getting loss and metric function handles')
        loss = getattr(module_loss, config['loss'])()
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        self.logger.debug('Initialising trainer')
        trainer = Trainer(model, loss, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler)

        trainer.train()
        self.logger.debug('Finished!')

    def _seed_everything(self, seed):
        self.logger.info(f'Using random seed: {seed}')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _resume_checkpoint(self, resume_path, rank, model, optimizer):
        """
        Resume from saved checkpoint.
        """
        if not resume_path:
            return model, optimizer

        self.logger.info(f'Loading checkpoint: {resume_path}')
        map_location = f'cuda:{rank}'
        checkpoint = torch.load(resume_path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from "
                                "that of checkpoint. Optimizer parameters not being resumed.")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

        amp.load_state_dict(checkpoint['amp'])
        self.logger.debug(f'Worker {rank} waiting to resume training')
        # dist.barrier()
        self.logger.info(f'Checkpoint "{resume_path}" loaded')
        return model, optimizer


if __name__ == '__main__':
    # Training settings
    config_filename = 'experiments/config.yml'
    config = load_config(config_filename)

    # Horovod: initialize library.
    hvd.init()
    torch.set_num_threads(8)

    # start
    Worker(config).train()
