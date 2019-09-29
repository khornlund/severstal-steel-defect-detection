import os
import random

from apex import amp
from apex.parallel import convert_syncbn_model  # noqa
# from apex.parallel import DistributedDataParallel as DPP
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DPP
# from torch.nn import SyncBatchNorm
from torch.multiprocessing.spawn import _wrap, SpawnContext
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


class Worker:

    @classmethod
    def spawn(cls, process_num, env, config, resume_filename):
        """
        Entry point for a new process to start training.

        Parameters
        ----------
        process_num : int
            Not used. Included to match function signature used by
            `torch.distributed.launch`.
        env : dict
            Environment state. Should include `LOCAL_RANK` and `WORLD_SIZE` settings.
        config : dict
            Config settings for training.
        resume_filename : str
            Path to model checkpoint to resume training (can be None).
        """
        os.environ = env
        config['local_rank'] = int(env['LOCAL_RANK'])
        config['world_size'] = int(env['WORLD_SIZE'])

        w = Worker(config)
        w.train(resume_filename)

    def __init__(self, config):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])
        self.config = config

    def train(self, resume):
        config = self.config
        rank = config['local_rank']
        world_size = config['world_size']
        self.logger.info(f'Worker {rank+1}/{world_size} starting...')

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)

        self.logger.debug(f'Cuda device count: {torch.cuda.device_count()}')
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', init_method='env://')

        model = model.to(device)
        self.logger.info(f'Using device {device}')

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(module_optimizer, 'optimizer', config, trainable_params)

        opt_level = config['apex']
        self.logger.debug(f'Setting apex opt_level: {opt_level}')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        model = convert_syncbn_model(model)

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        self.logger.info(f'Using `DistributedDataParallel`')
        model = DPP(model, device_ids=[device])
        model, optimizer = self._resume_checkpoint(resume, rank, model, optimizer)
        peek_weights = model.module.encoder._conv_stem.weight[0]
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
        dist.barrier()
        self.logger.info(f'Checkpoint "{resume_path}" loaded')
        return model, optimizer


class Master:
    """
    Custom implementation of `torch.distributed.launch`.
    """

    @classmethod
    def start(cls, config, resume_filename):
        master_addr    = config['distributed']['master_addr']
        master_port    = config['distributed']['master_port']
        n_nodes        = config['distributed']['n_nodes']
        node_rank      = config['distributed']['node_rank']
        nproc_per_node = config['distributed']['nproc_per_node']

        # world size in terms of number of processes
        dist_world_size = nproc_per_node * n_nodes

        # set PyTorch distributed related environmental variables
        current_env = os.environ.copy()
        current_env["MASTER_ADDR"]     = master_addr
        current_env["MASTER_PORT"]     = str(master_port)
        current_env["WORLD_SIZE"]      = str(dist_world_size)
        current_env["OMP_NUM_THREADS"] = str(1)

        ctx = mp.get_context('spawn')
        error_queues = []
        processes = []

        for local_rank in range(0, nproc_per_node):
            # each process's rank
            dist_rank = nproc_per_node * node_rank + local_rank
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            args = (current_env.copy(), config, resume_filename)

            # spawn the processes
            error_queue = ctx.SimpleQueue()
            process = ctx.Process(
                target=_wrap,
                args=(Worker.spawn, local_rank, args, error_queue)
            )
            process.start()
            error_queues.append(error_queue)
            processes.append(process)

        # Loop on join until it returns True or raises an exception.
        spawn_context = SpawnContext(processes, error_queues)
        while not spawn_context.join():
            pass
