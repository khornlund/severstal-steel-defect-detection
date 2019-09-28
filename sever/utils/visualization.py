try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError('Import `from torch.utils.tensorboard import SummaryWriter` failed.'
                      'Ensure PyTorch version >= 1.1 is installed.')


class TensorboardWriter():
    def __init__(self, writer_dir, enabled):
        self.writer = None
        if enabled:
            self.writer = SummaryWriter(writer_dir)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = [
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        ]
        self.tag_mode_exceptions = ['add_histogram', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = f'{self.mode}/{tag}'
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(f"type object `TensorboardWriter` has no attribute {name}")
            return attr
