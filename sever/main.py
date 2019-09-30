import os
import random

from apex import amp
import numpy as np
import torch
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


class Runner:

    def __init__(self, config):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])
        self.config = config

    def train(self, resume):
        config = self.config
        device_id = config['device']

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        device_ids = list(range(torch.cuda.device_count()))
        self.logger.debug(f'Using device {device_id} of {device_ids}')
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
        model = model.to(device)

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(module_optimizer, 'optimizer', config, trainable_params)

        opt_level = config['apex']
        self.logger.debug(f'Setting apex opt_level: {opt_level}')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        model, optimizer = self._resume_checkpoint(resume, model, optimizer)

        self.logger.debug('Getting augmentations')
        transforms = getattr(module_aug, config['augmentation'])()

        self.logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config, transforms)
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

    def _resume_checkpoint(self, resume_path, model, optimizer):
        """
        Resume from saved checkpoint.
        """
        if not resume_path:
            return model, optimizer

        self.logger.info(f'Loading checkpoint: {resume_path}')
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from "
                                "that of checkpoint. Optimizer parameters not being resumed.")
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])

        amp.load_state_dict(checkpoint['amp'])
        self.logger.info(f'Checkpoint "{resume_path}" loaded')
        return model, optimizer
