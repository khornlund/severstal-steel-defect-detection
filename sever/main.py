import os
import random

from apex import amp
# from apex.parallel import DistributedDataParallel
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DPP
import segmentation_models_pytorch as module_arch

import sever.data_loader.data_loaders as module_data
import sever.model.loss as module_loss
import sever.model.metric as module_metric
import sever.model.optimizer as module_optimizer
import sever.data_loader.augmentation as module_aug
from sever.trainer import Trainer
from sever.utils import setup_logger, setup_logging


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Master:

    @classmethod
    def start(self, config, resume_filename):
        world_size = config['world_size']
        mp.spawn(Worker.spawn,
                 args=(world_size, config, resume_filename),
                 nprocs=world_size,
                 join=True)


class Worker:

    @classmethod
    def spawn(cls, rank, world_size, config, resume_filename):
        w = Worker(rank, world_size, config)
        w.train(resume_filename)
        w.cleanup()

    def __init__(self, rank, world_size, config):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        config['rank'] = rank
        config['world_size'] = world_size
        self.config = config

    def train(self, resume):
        config = self.config
        rank = config['rank']
        world_size = config['world_size']

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        self.logger.info(f'Using device {device}')

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(module_optimizer, 'optimizer', config, trainable_params)

        opt_level = config['apex']
        self.logger.debug(f'Setting apex opt_level: {opt_level}')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        self.logger.info(f'Using `DistributedDataParallel`')
        model = DPP(model, device_ids=[device])

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
                        resume=resume,
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

    def cleanup(self):
        self.logger.info('Cleanup: destroying process group')
        dist.destroy_process_group()
