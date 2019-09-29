import numpy as np
import torch
from apex import amp
from torchvision.utils import make_grid

from sever.base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config, device)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 4

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        self.writer.set_step((epoch - 1) * len(self.data_loader))
        for idx, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('LR', param_group['lr'])

        losses = AverageMeter('loss')
        metrics = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

                losses.update(loss.item(), data.size(0))
                self.writer.add_scalar('loss', loss.item())
                for i, value in enumerate(self._eval_metrics(output, target)):
                    metrics[i].update(value, data.size(0))
                    self.writer.add_scalar(metrics[i].name, value)

                self._log_batch(epoch, batch_idx, self.data_loader.batch_size,
                                len(self.data_loader), loss.item())

            if batch_idx == 1:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=2, normalize=True))

        log = {
            'loss': losses.avg,
            'metrics': [m.avg for m in metrics]
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _log_batch(self, epoch, batch_idx, batch_size, len_data, loss):
        n_samples = batch_size * len_data
        n_complete = batch_idx * batch_size
        percent = 100.0 * batch_idx / len_data
        msg = f'Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] Loss: {loss:.6f}'
        self.logger.debug(msg)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        losses = AverageMeter('loss')
        metrics = [AverageMeter(m.__name__) for m in self.metrics]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                losses.update(loss.item(), data.size(0))
                for i, value in enumerate(self._eval_metrics(output, target)):
                    metrics[i].update(value, data.size(0))

        self.writer.set_step((epoch - 1), 'valid')
        self.writer.add_scalar('loss', losses.avg)
        for m in metrics:
            self.writer.add_scalar(m.name, m.avg)

        return {
            'val_loss': losses.avg,
            'val_metrics': [m.avg for m in metrics]
        }

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            for i, metric in enumerate(self.metrics):
                value = metric(output, target)
                yield value


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
