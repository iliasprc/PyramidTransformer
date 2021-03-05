# Sign language pyramid transformer PT



* [Sign language pyramid transformer](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Training](#training)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
        * [Using Multiple GPU](#using-multiple-gpu)
		* [Testing](#testing)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
	* [Pretrained Models](#pretrained-models)

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
* tensorboard >= 1.14 
* scikit-video
* av
## Features


## Folder Structure
  ```
  PyramidTransformer/
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
  │   ├── trainer_RGBD_config.yml
  │   └── 
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
  ├── trainer/ - training and testing functions
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
Try `python train.py -c config.yml` to run code.

### Config file format
Config files are in `.yml` format. These are the default training options
```yaml
ttrainer:
  input_data: /home/papastrat/Desktop/ilias/datasets/ # path of input data
  cwd: /home/papastrat/PycharmProjects/PyramidTransformer # working directory
  logger: SLR_challenge # logger name
  epochs: 30 # number of training epochs
  seed: 1234 # randomness seed
  cuda: True # use nvidia gpu
  gpu: 0,1 # id of gpu
  save: True # save checkpoint
  load: False # load pretrained checkpoint
  gradient_accumulation: 16 # gradient accumulation steps
  pretrained_cpkt: best_model.pth
  log_interval: 1000 # print statistics every log_interval
  model:
    name: IR_CSN_152 # model name
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
      num_workers: 4 # number of thread for dataloader
    validation:
      batch_size: 4
      shuffle: False
      num_workers: 4
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


### Training
To train the network simplt run:

  ```
  python train.py 
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --load --cpkt path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --gpu 0,1 -c config.yml
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=0,1 python train.py -c config.yml
  ```


### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

```
python test.py  -c config.yml -cpkt path/to/checkpoint
```

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`




## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
