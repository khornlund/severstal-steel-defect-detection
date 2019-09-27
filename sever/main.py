import os
import random

from apex import amp
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as module_arch

import sever.data_loader.data_loaders as module_data
import sever.model.loss as module_loss
import sever.model.metric as module_metric
import sever.model.optimizer as module_optimizer
import sever.data_loader.augmentation as module_aug
from sever.trainer import Trainer
from sever.utils import setup_logger, setup_logging
from sever.data_loader import PostProcessor, mask2rle


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Runner:

    def train(self, config, resume):
        setup_logging(config)
        self.logger = setup_logger(self, config['training']['verbose'])
        self._seed_everything(config['seed'])

        self.logger.debug('Getting augmentations')
        transforms = getattr(module_aug, config['augmentation'])()

        self.logger.debug('Getting data_loader instance')
        data_loader = get_instance(module_data, 'data_loader', config, transforms)
        valid_data_loader = data_loader.split_validation()

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)
        model, device = self._prepare_device(model, config['n_gpu'])

        self.logger.debug('Freezing encoder weights')
        for p in model.encoder.parameters():
            p.requires_grad = False

        self.logger.debug('Getting loss and metric function handles')
        loss = getattr(module_loss, config['loss'])()
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        self.logger.debug('Building optimizer and lr scheduler')
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = get_instance(module_optimizer, 'optimizer', config, trainable_params)

        opt_level = config['apex']
        self.logger.debug(f'Setting apex opt_level: {opt_level}')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler',
                                    config, optimizer)

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

    def predict(self, config, model_checkpoint):
        setup_logging(config)
        self.logger = setup_logger(self, config['testing']['verbose'])
        self._seed_everything(config['seed'])

        self.logger.info(f'Using config:\n{config}')

        self.logger.debug('Getting data_loader instance')
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['testing']['data_dir'],
            batch_size=config['testing']['batch_size'],
            shuffle=False,
            validation_split=0.0,
            train=False,
            nworkers=config['testing']['nworkers'],
            verbose=config['testing']['verbose']
        )

        self.logger.debug('Building model architecture')
        model = get_instance(module_arch, 'arch', config)
        model, device = self._prepare_device(model, config['n_gpu'])

        opt_level = config['apex']
        model = amp.initialize(model, opt_level=opt_level)

        self.logger.debug(f'Loading checkpoint {model_checkpoint}')
        checkpoint = torch.load(model_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        amp.load_state_dict(checkpoint['amp'])

        pp = PostProcessor()

        model.eval()
        predictions = []

        bs = data_loader.batch_size
        preds = torch.zeros(len(data_loader) * bs, 4, 256, 1600)  # N may be too large
        ids = []
        self.logger.debug('Starting...')
        with torch.no_grad():
            for bidx, (batch_ids, data) in enumerate(tqdm(data_loader)):
                ids.extend(batch_ids)
                data = data.to(device)
                output = model(data)
                output_size = output.size()
                batch_preds = torch.sigmoid(output).detach().cpu().numpy()
                preds[bidx * bs:(bidx + 1) * output_size[0], :, :, :] = batch_preds

        for (f, p) in zip(ids, preds):
            for class_, pred in enumerate(p):
                pred, num = pp.process(class_, pred)
                rle = mask2rle(pred)
                name = f + f"_{class_+1}"
                predictions.append([name, rle])

        # save predictions to submission.csv
        df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
        self.logger.info(df.head(20))
        df.to_csv("submission.csv", index=False)
        self.logger.info(f'Finished saving predictions to "submission.csv"')

    def _prepare_device(self, model, n_gpu_use):
        device, device_ids = self._get_device(n_gpu_use)
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model, device

    def _get_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, "
                                f"but only {n_gpu} are available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        self.logger.info(f'Using device: {device}, {list_ids}')
        return device, list_ids

    def _seed_everything(self, seed):
        self.logger.info(f'Using random seed: {seed}')
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
