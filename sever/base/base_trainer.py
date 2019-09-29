import os
import math
import shutil

import yaml
import torch
import torch.distributed as dist
from apex import amp

from sever.utils import setup_logger, trainer_paths, TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config, device):
        self.logger = setup_logger(self, verbose=config['training']['verbose'])
        self.model = model
        self.device = device
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config
        self.rank = config['local_rank']
        self.world_size = config['world_size']

        cfg_trainer = config['training']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        self.checkpoint_dir, writer_dir = trainer_paths(config)
        # setup visualization writer instance
        tb_enabled = (cfg_trainer['tensorboard'] and (self.rank == 0))
        self.writer = TensorboardWriter(writer_dir, tb_enabled)

        # Save configuration file into checkpoint directory:
        config_save_path = os.path.join(self.checkpoint_dir, 'config.yaml')
        with open(config_save_path, 'w') as handle:
            yaml.dump(config, handle, default_flow_style=False)

    def train(self):
        """
        Full training logic
        """
        self.logger.info('Starting training...')
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({
                        mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({
                        'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info(f'{str(key):15s}: {value}')

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according
                    # to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or\
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Warning: Metric '{self.mnt_metric}' is not found. Model "
                                        "performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for "
                                     f"{self.early_stop} epochs. Training stops.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        self.logger.debug(f'Trainer {self.rank} waiting at save point')
        dist.barrier()  # sync
        if self.rank != 0:
            return

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'amp': amp.state_dict()
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Trainer {self.rank} saving checkpoint: {filename} ...")
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            self.logger.info(f'Saving current best: {best_path}')
            shutil.copyfile(filename, best_path)

    def _reduce_tensor(self, t):
        rt = t.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt
