=====
sever
=====
PyTorch deep learning project made easy.

.. contents:: Table of Contents
   :depth: 2

Submission Log
==============

+-------+-------+-----------+------+-----------+--------+---------------------+
| Score | Model | Optimizer |  LR  | Scheduler | Epochs |        Notes        |
+=======+=======+===========+======+===========+========+=====================+
| 87975 |  B4   |   RAdam   | 1e-3 |    Step   |    2   | First sub.          |
+-------+-------+-----------+------+-----------+--------+---------------------+

Requirements
============
* Python >= 3.6
* PyTorch >= 1.1
* Tensorboard >= 1.4

Folder Structure
================

::

  cookiecutter-pytorch/
  │
  ├── <project name>/
  │    │
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── base/ - abstract base classes
  │    │   ├── base_data_loader.py - abstract base class for data loaders
  │    │   ├── base_model.py - abstract base class for models
  │    │   └── base_trainer.py - abstract base class for trainers
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │   └── data_loaders.py
  │    │
  │    ├── model/ - models, losses, and metrics
  │    │   ├── loss.py
  │    │   ├── metric.py
  │    │   └── model.py
  │    │
  │    ├── trainer/ - trainers
  │    │   └── trainer.py
  │    │
  │    └── utils/
  │        ├── logger.py - class for train logging
  │        ├── visualization.py - class for Tensorboard visualization support
  │        └── saving.py - manages pathing for saving models + logs
  │
  ├── logging.yml - logging configuration
  │
  ├── data/ - directory for storing input data
  │
  ├── experiments/ - directory for storing configuration files
  │
  ├── saved/ - directory for checkpoints and logs
  │
  └── tests/ - tests folder


Usage
=====

.. code-block:: bash

  $ conda env create --file environment.yml
  $ conda activate sever

The code in this repo is an MNIST example of the template. You can run the tests,
and the example project using:

.. code-block:: bash

  $ pytest tests
  $ sever train -c experiments/config.yml

Config file format
------------------
Config files are in `.yml` format:

.. code-block:: HTML

  short_name: Mnist_LeNet
  n_gpu: 1
  save_dir: saved/
  seed: 1234

  arch:
    type: MnistModel
    args:
      verbose: 2

  data_loader:
    type: MnistDataLoader
    args:
      batch_size: 128
      data_dir: data/
      num_workers: 2
      shuffle: true
      validation_split: 0.1

  loss: nll_loss

  lr_scheduler:
    type: StepLR
    args:
      gamma: 0.1
      step_size: 50

  metrics:
  - my_metric
  - my_metric2

  optimizer:
    type: Adam
    args:
      lr: 0.001
      weight_decay: 0

  training:
    early_stop: 10
    epochs: 100
    monitor: min val_loss
    save_period: 1
    tensorboard: true
    verbose: 2

  testing:
    data_dir: data/
    batch_size: 128
    num_workers: 8
    verbose: 2


Add addional configurations if you need.

Using config files
------------------
Modify the configurations in `.yml` config files, then run:

.. code-block:: shell

  sever train -c experiments/config.yml

Resuming from checkpoints
-------------------------
You can resume from a previously saved checkpoint by:

.. code-block:: shell

  sever train --resume path/to/checkpoint


Using Multiple GPU
------------------
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.

.. code-block:: shell

  sever train --device 2,3 -c experiments/config.yml


Customization
=============

Data Loader
-----------

Writing your own data loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherit `BaseDataLoader`
^^^^^^^^^^^^^^^^^^^^^^^^
`BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

`BaseDataLoader` handles:
* Generating next batch
* Data shuffling
* Generating validation data loader by calling
`BaseDataLoader.split_validation()`

DataLoader Usage
~~~~~~~~~~~~~~~~
`BaseDataLoader` is an iterator, to iterate through batches:

.. code-block:: python

  for batch_idx, (x_batch, y_batch) in data_loader:
      pass

Example
~~~~~~~
Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

Trainer
-------

Writing your own trainer
~~~~~~~~~~~~~~~~~~~~~~~~

Inherit `BaseTrainer`
^^^^^^^^^^^^^^^^^^^^^

`BaseTrainer` handles:
1. Training process logging
2. Checkpoint saving
3. Checkpoint resuming
4. Reconfigurable performance monitoring for saving current best model, and early stop training.

  1. If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a
      checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
  2. If config `early_stop` is set, training will be automatically terminated when model
      performance does not improve for given number of epochs. This feature can be turned off by
      passing 0 to the `early_stop` option, or just deleting the line of config.

Implementing abstract methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need to implement `_train_epoch()` for your training process, if you need validation then
you can implement `_valid_epoch()` as in `trainer/trainer.py`

Example
~~~~~~~
Please refer to `trainer/trainer.py` for MNIST training.

Model
-----

Writing your own model
~~~~~~~~~~~~~~~~~~~~~~

Inherit `BaseModel`
^^^^^^^^^^^^^^^^^^^
`BaseModel` handles:
  * Inherited from `torch.nn.Module`
  * `__str__`: Modify native `print` function to prints the number of trainable parameters.

Implementing abstract methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implement the foward pass method `forward()`

Example
~~~~~~~
Please refer to `model/model.py` for a LeNet example.

Loss
----
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in
"loss" in config file, to corresponding name.

Metrics
~~~~~~~
Metric functions are located in `model/metric.py`.

You can monitor multiple metrics by providing a list in the configuration file, eg.

.. code-block:: HTML

  "metrics": ["my_metric", "my_metric2"]


Additional logging
------------------
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge
them with `log` as shown below before returning:

.. code-block:: python

  additional_log = {"gradient_norm": g, "sensitivity": s}
  log = {**log, **additional_log}
  return log

Testing
-------
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume`
argument.

Validation data
---------------
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, it will
return a validation data loader, with the number of samples according to the specified ratio in your
config file.

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

Checkpoints
-----------
You can specify the name of the training session in config files:

.. code-block:: HTML

  "name": "MNIST_LeNet"


The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in
mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:

.. code-block:: python

  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }


Tensorboard Visualization
--------------------------
This template supports `<https://pytorch.org/docs/stable/tensorboard.html>`_ visualization.

1. Run training

    Set `tensorboard` option in config file true.

2. Open tensorboard server

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at
    `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of
model parameters will be logged. If you need more visualizations, use `add_scalar('tag', data)`,
`add_image('tag', image)`, etc in the `trainer._train_epoch` method. `add_something()` methods in
this template are basically wrappers for those of `tensorboard.SummaryWriter` module.

**Note**: You don't have to specify current steps, since `TensorboardWriter` class defined at
`logger/visualization.py` will track current steps.

Acknowledgments
===============
This template is inspired by

  1. `<https://github.com/victoresque/pytorch-template>`_
  2. `<https://github.com/daemonslayer/cookiecutter-pytorch>`_
