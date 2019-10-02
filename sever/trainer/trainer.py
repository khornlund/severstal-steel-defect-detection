import numpy as np
import torch
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
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 16
        self.unfreeze_encoder = config['training']['unfreeze_encoder']

        self.logger.info('Freezing encoder weights')
        for p in self.model.encoder.parameters():
            p.requires_grad = False

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
        if self.unfreeze_encoder is not None and epoch >= self.unfreeze_encoder:
            self.logger.info('Unfreezing encoder weights')
            for p in self.model.encoder.parameters():
                p.requires_grad = True
            self.unfreeze_encoder = None

        self.model.train()
        self.writer.set_step((epoch) * len(self.data_loader))
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:
                self.writer.add_scalar('LR/encoder', param_group['lr'])
            elif i == 1:
                self.writer.add_scalar('LR/decoder', param_group['lr'])

        losses_comb = AverageMeter('loss_comb')
        losses_bce  = AverageMeter('loss_bce')
        losses_dice = AverageMeter('loss_dice')
        metrics = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # self.logger.info(f'output: {output.size()}')
            # self.logger.info(f'target: {target.size()}')
            loss, bce, dice = self.loss(output, target)

            loss.backward()
            self.optimizer.step()

            losses_comb.update(loss.item(), data.size(0))
            losses_bce.update(bce.item(),   data.size(0))
            losses_dice.update(dice.item(), data.size(0))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch) * len(self.data_loader) + batch_idx)

                self.writer.add_scalar('batch/loss', loss.item())
                self.writer.add_scalar('batch/bce',  bce.item())
                self.writer.add_scalar('batch/dice', dice.item())

                for i, value in enumerate(self._eval_metrics(output, target)):
                    metrics[i].update(value, data.size(0))
                    self.writer.add_scalar(f'batch/{metrics[i].name}', value)

                self._log_batch(epoch, batch_idx, self.data_loader.batch_size,
                                len(self.data_loader), loss.item())

            if batch_idx == 0:
                with torch.no_grad():
                    data = data.cpu()
                    target = torch.max(target, dim=1, keepdim=True)[0].cpu()
                    output = torch.max(output, dim=1, keepdim=True)[0].cpu()

                    truth = torch.cat([data, target, target], dim=1)
                    preds = torch.cat([data, output, output], dim=1)
                    self.writer.add_image('truth', make_grid(truth, nrow=8, normalize=True))
                    self.writer.add_image('preds', make_grid(preds, nrow=8, normalize=True))

        self.writer.add_scalar('epoch/loss', losses_comb.avg)
        self.writer.add_scalar('epoch/bce',  losses_bce.avg)
        self.writer.add_scalar('epoch/dice', losses_dice.avg)
        for m in metrics:
            self.writer.add_scalar(f'epoch/{m.name}', m.avg)

        del data
        del target
        del output

        log = {
            'loss': losses_comb.avg,
            'metrics': [m.avg for m in metrics]
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

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
        losses_comb = AverageMeter('loss_comb')
        losses_bce  = AverageMeter('loss_bce')
        losses_dice = AverageMeter('loss_dice')
        metrics = [AverageMeter(m.__name__) for m in self.metrics]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss, bce, dice = self.loss(output, target)

                losses_comb.update(loss.item(), data.size(0))
                losses_bce.update(bce.item(),   data.size(0))
                losses_dice.update(dice.item(), data.size(0))

                for i, value in enumerate(self._eval_metrics(output, target)):
                    metrics[i].update(value, data.size(0))

        self.writer.set_step((epoch), 'valid')
        self.writer.add_scalar('loss', losses_comb.avg)
        self.writer.add_scalar('bce', losses_bce.avg)
        self.writer.add_scalar('dice', losses_dice.avg)
        for m in metrics:
            self.writer.add_scalar(m.name, m.avg)

        return {
            'val_loss': losses_comb.avg,
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
