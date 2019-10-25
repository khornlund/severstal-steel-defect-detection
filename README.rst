=========================================================================================================
`Severstal: Steel Defect Detection <https://www.kaggle.com/c/severstal-steel-defect-detection/overview>`_
=========================================================================================================

*Severstal is leading the charge in efficient steel mining and production. The company recently
created the country’s largest industrial data lake, with petabytes of data that were previously
discarded. Severstal is now looking to machine learning to improve automation, increase efficiency,
and maintain high quality in their production.

In this competition, you’ll help engineers improve the algorithm by localizing and classifying
surface defects on a steel sheet.*

.. contents::
   :depth: 2


Competition Report
==================

Results
-------
Winning submission:

+-----------+------------+
| Public LB | Private LB |
+===========+============+
|  0.92124  |  0.90883   |
+-----------+------------+

My best submission:

+-----------+------------+
| Public LB | Private LB |
+===========+============+
|  0.91817  |  0.91023   |
+-----------+------------+

My chosen submission:

+-----------+------------+
| Public LB | Private LB |
+===========+============+
|  0.91844  |   0.90274  |
+-----------+------------+

I made some poor choices choosing my submission, and ended up rank 55/2436.

Models
------
I used `segmentation_models.pytorch <https://github.com/qubvel/segmentation_models.pytorch>`_ (SMP)
as a framework for all of my models. It's a really nice package and easy to extend, so I implemented
a few of my own encoder and decoder modules.

I used an ensemble of models, covered below.

Encoders
~~~~~~~~
I ported `EfficientNet <https://github.com/lukemelas/EfficientNet-PyTorch>`_ to the above framework
and had great results. I was hoping this would be a competitive advantage, but during the
competition someone added an EfficientNet encoder to SMP. I used the `b5` model for most of the
competition, and found the smaller models didn't work as well.

I also ported ``InceptionV4`` late in the competition and had pretty good results.

I ported a few others that didn't yield good results:
    - `Res2Net <https://github.com/gasvn/Res2Net>`_
    - `Dilated ResNet <https://github.com/wuhuikai/FastFCN/blob/master/encoding/dilated/resnet.py>`_

I had good results using ``se_resnext50_32x4d`` too. I found that because it didn't consume as much
memory as the `efficientnet-b5`, I could use larger batch and image sizes which led to improvements.

Decoders
~~~~~~~~
I used Unet + FPN from SMP.

I implemented `Nested Unet <https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py>`_
such that it could use pretrained encoders, but it didn't yield good results.

Other
~~~~~
I ported `DeepLabV3 <https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py>`_
to SMP but didn't get good results.

Scores
~~~~~~
These are the highest (private) scoring single models of each architecture.

+--------------------+---------+-----------+------------+
|       Encoder      | Decoder | Public LB | Private LB |
+====================+=========+===========+============+
|  efficientnet-b5   |    FPN  |  0.91631  |   0.90110  |
+--------------------+---------+-----------+------------+
|  efficientnet-b5   |   Unet  |  0.91665  |   0.89769  |
+--------------------+---------+-----------+------------+
| se_resnext50_32x4d |    FPN  |  0.91744  |   0.90038  |
+--------------------+---------+-----------+------------+
| se_resnext50_32x4d |   Unet  |  0.91685  |   0.89647  |
+--------------------+---------+-----------+------------+
|    inceptionv4     |    FPN  |  0.91667  |   0.89149  |
+--------------------+---------+-----------+------------+

Training
--------

Loss
~~~~
I used (0.6 * BCE) + (0.4 * (1 - Dice)). I applied smoothing (1e-6) to the labels.

Optimizer
~~~~~~~~~
RAdam

Encoder learning rate 7e-5
Decoder learning rate 3e-3

LR Schedule
~~~~~~~~~~~
Flat for 30 epochs, then cosine anneal over 220 epochs. Typically I stopped training around 150-200
epochs.

Image Sizes
~~~~~~~~~~~
256x384, 256x416, 256x448, 256x480

Larger image sizes gave better results, but so did larger batch sizes. The ``se_resnext50_32x4d``
encoders could use a batch size of 32-46, while the ``efficientnet-b5`` encoders typically used a
batch size of 16-20.

Grayscale Input
~~~~~~~~~~~~~~~
The images were provided as 3-channel grayscale. I modified the models to accept 1 channel input,
by recycling pretrained weights. I did a bunch of testing around this as I was worried it might
hurt performance, but using 3-channel input didn't give better results.

I parameterised the recycling of the weights so I could train models using the R, G, or B pretrained
weights for the first conv layer. My hope was that this would produce a more diverse model ensemble.

Augmentation
~~~~~~~~~~~~
I used the following `Albumentations <https://github.com/albu/albumentations>`_:

.. code::

    Compose([
        OneOf([
            CropNonEmptyMaskIfExists(self.height, self.width),
            RandomCrop(self.height, self.width)
        ], p=1),
        OneOf([
            CLAHE(p=0.5),  # modified source to get this to work with grayscale
            GaussianBlur(3, p=0.3),
            IAASharpen(alpha=(0.2, 0.3), p=0.3),
        ], p=1),
        Flip(p=0.5),
        Normalize(mean=[0.3439], std=[0.0383]),
        ToTensor(),
    ])

It would have been nice to experiment with more of these, but it took so long to train the models
it was difficult. I found these augs worked better than simple crops/flips and stuck with them.

Validation
~~~~~~~~~~
I used a random 20% of the training data for validation with each run.

Pseudo Labels
~~~~~~~~~~~~~
I used the ensemble outputs of models as pseudo labels, which gave a huge performance boost. I
used a custom `BatchSampler <https://github.com/khornlund/pytorch-balanced-sampler>`_ to undersample
(sample rate ~60%) from the pseudo-labelled data, and fix the number of pseudo-labelled samples per
batch (each batch would contain 12% pseudo-labelled samples).

Some other people had poor results with pseudo-labels. Perhaps the technique above helped mitigate
whatever downsides they faced.




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
