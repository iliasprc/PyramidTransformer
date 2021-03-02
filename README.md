# Sign language pyramid transformer PT



* [Sign language pyramid transformer](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements

```python
pip install -r requirements.txt
```

* Python >= 3.5 (3.6 recommended)
* PyTorch >= 1.4 (1.6.0 recommended)
* torchvision >=0.6.0  
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* scikit-video
* av
## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  ├── checkpoints/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output

  │
  ├── config - holds configuration for training
  │   ├── trainer_config.yml
  │   ├── 
  │   └── 
  │
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── loader_utils.py - functions to load and preprocess data
  │   ├── dataset.py - initialize dataloader functions
  ├── data/ - default directory for storing input data
  │
  ├── models/ - models, losses, and metrics
  │   ├── model.py
  │   ├── model_utils.py - model functions, optimizer and weight initializations
  │
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── metrics.py
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── train_RGBD.py - main script to start training on RGBD data
  ├── test_RGBD.py - evaluation of trained model  on RGBD data
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.yaml` format:
```yaml
trainer:
  input_data: /home/papastrat/Desktop/ilias/datasets/ # path of input data
  cwd: /home/papastrat/PycharmProjects/SLR_challenge # working directory
  logger: SLR_challenge # logger name
  epochs: 10 # number of training epochs
  seed: 1234 # randomness seed
  cuda: True # use nvidia gpu
  gpu: 0 # id of gpu
  save: True # save checkpoint
  load: False # load pretrained checkpoint
  gradient_accumulation: 16 # gradient accumulation steps
  pretrained_cpkt: /home/papastrat/PycharmProjects/SLR_challenge/checkpoints/model_IR_CSN_152/dataset_Autsl/date_25_02_2021_09.24.20/best_model.pth
  log_interval: 1000 # print statistics every log_interval
  model:
    name: Pyramid_Transformer # model name
    optimizer: # optimizer configuration
      type: SGD # optimizer type
      lr: 1e-2 # learning rate
      weight_decay: 0.000001 # weight decay
    scheduler: # learning rate scheduler
      type: ReduceLRonPlateau # type of scheduler
      scheduler_factor: 0.5 # learning rate change ratio
      scheduler_patience: 0 # patience for some epochs
      scheduler_min_lr: 1e-3 # minimum learning rate value
      scheduler_verbose: 5e-6 # print if learning rate is changed
  dataloader:
    train:
      batch_size: 4 # batch size
      shuffle: True # shuffle samples after every epoch
      num_workers: 2 # number of thread for dataloader
    validation:
      batch_size: 4
      shuffle: False
      num_workers: 2
    test:
      batch_size: 1
      shuffle: False
      num_workers: 2
  dataset:
    input_data: /home/papastrat/Desktop/ilias/datasets/
    name: AUTSL # dataset name
    modality: RGB # type of modality
    images_path: challenge/ # data directory
    classes: 226 # number of classes
    normalize: True # mean, std normalization
    padding : True # pad videos
    dim: [224,224] # frame dimension
    train:
      seq_length: 64 # number of frames for each video
      augmentation: True # do augmentation to video
    val:
      seq_length: 64
      augmentation: False
    validation:
      seq_length: 64
      augmentation: False
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py 
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization



### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`


### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Multiple optimizers
- [ ] Support more tensorboard functions
- [x] Using fixed random seed
- [x] Support pytorch native tensorboard
- [x] `tensorboardX` logger support
- [x] Configurable logging layout, checkpoint naming
- [x] Iteration-based training (instead of epoch-based)
- [x] Adding command line option for fine-tuning

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
