import os
import sys
import subprocess
import random
import multiprocessing as mp
from argparse import ArgumentParser, REMAINDER

from apex import amp
from apex.parallel import DistributedDataParallel as DPP
import numpy as np
import torch
# import torch.multiprocessing as mp
import torch.distributed as dist
# from torch.distributed import launch
# from torch.nn.parallel import DistributedDataParallel as DPP
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
    def spawn(cls, env, config, resume_filename):
        # print(f'CURRENT ENV: {os.environ}')
        # print(f'RECEIVED ENV: {env}')
        os.environ = env
        config['local_rank'] = int(env['LOCAL_RANK'])
        config['world_size'] = int(env['WORLD_SIZE'])

        w = Worker(config)
        w.train(resume_filename)
        w.cleanup()

    def __init__(self, config):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])
        self.config = config

    def train(self, resume):
        config = self.config
        rank = config['local_rank']
        world_size = config['world_size']
        self.logger.info(f'Worker {rank}/{world_size} starting...')

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

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

        self.logger.info(f'Using `DistributedDataParallel`')
        model = DPP(model)

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


class Master:
    """
    Custom implementation of `torch.distributed.launch`
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
        current_env["MASTER_ADDR"] = master_addr
        current_env["MASTER_PORT"] = str(master_port)
        current_env["WORLD_SIZE"]  = str(dist_world_size)

        processes = []
        mp.set_start_method('spawn')

        if 'OMP_NUM_THREADS' not in os.environ and nproc_per_node > 1:
            current_env["OMP_NUM_THREADS"] = str(1)
            print("*****************************************\n"
                "Setting OMP_NUM_THREADS environment variable for each process "
                "to be {} in default, to avoid your system being overloaded, "
                "please further tune the variable for optimal performance in "
                "your application as needed. \n"
                "*****************************************".format(current_env["OMP_NUM_THREADS"]))

        for local_rank in range(0, nproc_per_node):
            # each process's rank
            dist_rank = nproc_per_node * node_rank + local_rank
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            p = mp.Process(target=Worker.spawn, name=f'worker-rank-{local_rank}', kwargs={
                'env': current_env,
                'config': config,
                'resume_filename': resume_filename
            })
            p.start()
            processes.append(p)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise Exception(f'Process {process.name} returned code {process.exitcode}')
