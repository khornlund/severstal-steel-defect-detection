import os
import yaml
import logging.config

from .saving import log_path


def setup_logging(run_config, log_config='logging.yml', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    if os.path.exists(log_config):
        with open(log_config, 'rt') as f:
            config = yaml.safe_load(f.read())

        # modify logging paths based on run config
        run_path = log_path(run_config)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = os.path.join(run_path, handler['filename'])

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


def setup_logger(cls, verbose=0):
    logger = logging.getLogger(cls.__class__.__name__)
    if verbose not in logging_level_dict:
        raise KeyError(f'verbose option {verbose} for {cls} not valid. '
                       f'Valid options are {logging_level_dict.keys()}.')
    logger.setLevel(logging_level_dict[verbose])
    return logger
