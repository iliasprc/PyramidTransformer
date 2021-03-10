# Sign Language Video Pyramid Transformer (SLVPT)
## Abstract

## Contents
* [Sign Language Video Pyramid Transformer](#pytorch-template-project)
    * [Abstract](#abstract)
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






## TODOs

- [ ] Final Requirements
- [ ] Pretrained models
- [ ] Test all pretrained models
- [ ] Instructions for training 
- [ ] Adding command line option for inference

## Requirements

### Installation & Data Preparation

Please refer to 
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
  │   	  └── log/ - default logdir for  logging output
  │
  ├── config - holds configuration for training
  |   ├── RGB/ -configs for training and testing with  RGB modality
  │   ├── Depth/ -configs for training and testing with depth modality 
  │   ├── RGBD/
  │   ├── trainer_config.yml
  │   ├── trainer_RGBD_config.yml
  │   └── 
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── autsl / 
  │       ├── autsl_loader - dataloader for AUTSL dataset
  │   ├── loader_utils.py - functions to load and preprocess data
  │   ├── dataset.py - initialize dataloader functions
  ├── data/ - default directory for storing input data
  │
  ├── models/ - models, losses, and metrics
  │   ├── model.py
  │   ├── model_utils.py - model functions, optimizer and weight initializations
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


### Config file format
Config files are in `.yml` format. These are the default training options
```yaml
trainer:
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
    val:
      batch_size: 4
      shuffle: False
      num_workers: 4
    test:
      batch_size: 1
      shuffle: False
      num_workers: 2
  dataset:
    input_data: data/
    name: AUTSL # dataset name
    modality: depth # type of modality
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
    test:
      seq_length: 64
      augmentation: False
```


### Training
To train any network simply run:

  ```
  python train.py -c path/to/config.yml
  ```

#### RGB modality
To train the  Pyramid Transformer using RGB modality, initially train the 3D backbone as:


  ```
  python train.py -c config/RGB/IR_CSN_152/trainer_config.yml
  ```
Then, using the trained 3D-CNN backbone train the Pyramid Transformer:

  ```
  python train.py -c config/RGB/Pyramid_Transformer/trainer_config.yml --pretrained_cpt path/to/backbone.pth
  ```


#### Depth modality

To train the  Pyramid Transformer using Depth modality, initially train the 3D backbone as:


  ```
  python train.py -c config/Depth/IR_CSN_152/trainer_config.yml
  ```
Then, using the trained 3D-CNN backbone train the Pyramid Transformer:

  ```
  python train.py -c config/Depth/Pyramid_Transformer/trainer_config.yml --pretrained_cpkt path/to/backbone.pth
  ```


#### RGBD modality

To train the  RGBD Transformer using RGB modality:


  ```
  python train_RGBD.py -c config/RGBD/trainer_RGBD_config.yml 
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
  python train.py --gpu 0,1 -c path/to/config.yml
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=0,1 python train.py -c path/to/config.yml
  ```


### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--pretrained_cpkt` argument.

```
python test.py  -c path/to/config.yml --pretrained_cpkt path/to/checkpoint
```

## Pretrained Models

Link to google drive with pretrained models

[RGB Pyramid Transformer](linkrgb)

[RGBD Pyramid Transformer](linkrgb)


| Model                                                    | Backbone | Pretrain     | #Frame | Param. |  GFLOPs |   Validation Acc (%) |    Test Acc (%)      |                                                        Weights                                                               |
|----------------------------------------------------------|:------------:|:--------:|:------:|:------:|:-------:|:--------------------:|:--------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|
|[RGB Video Pyramid Transformer]()                         |[ir-CSN-152]()| [**IG-65M**](https://research.fb.com/wp-content/uploads/2019/05/Large-scale-weakly-supervised-pre-training-for-video-action-recognition.pdf)    | 64     |        |          |                      |                     |                                                                          |
|[RGBD Video  Pyramid Transformer]()                       |[ir-CSN-152]()| [**IG-65M**](https://research.fb.com/wp-content/uploads/2019/05/Large-scale-weakly-supervised-pre-training-for-video-action-recognition.pdf) | 64     |        |          |                      |                     |                                                                          |

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
We appreciate developers of Video Model Zoo [VMZ](https://github.com/facebookresearch/VMZ) for the pretrained 3D-CNN models