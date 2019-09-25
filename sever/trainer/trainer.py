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
    def __init__(self, model, loss, metrics, optimizer, resume, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, resume, config, device)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            acc_metrics = np.zeros(len(self.metrics) + 1)
            for i, metric in enumerate(self.metrics):
                acc_metrics[i] += metric(output, target)
                self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
            return acc_metrics

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
        self.writer.add_scalar('LR', self._get_lr())

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            # self.logger.debug(f'data: {data.size()}, target: {target.size()}')
            self.optimizer.zero_grad()
            output = self.model(data)
            # self.logger.debug(f'output: {output.size()}')
            loss = self.loss(output, target)
            # loss.backward()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self._log_batch(epoch, batch_idx, self.data_loader.batch_size,
                                self.data_loader.n_samples, len(self.data_loader), loss.item())
            if batch_idx == 1:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # self.writer.add_image('target', make_grid(target.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
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

    def _log_batch(self, epoch, batch_idx, batch_size, n_samples, len_data, loss):
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
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

        total_val_loss /= len(self.valid_data_loader)
        self.writer.set_step((epoch - 1), 'valid')
        self.writer.add_scalar('loss', total_val_loss)

        return {
            'val_loss': total_val_loss,
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
