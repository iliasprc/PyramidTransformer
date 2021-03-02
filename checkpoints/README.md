# Checkpoints 

Model's checkpoints are stored in this folder

Each checkpoint folder has the name
```python
'checkpoints/model_' + config.model.name + '/dataset_' + config.dataset.name + '/date_' + now.strftime("%d_%m_%Y_%H.%M.%S"))
```
e.g.,  

```python

'./checkpoints/model_Pyramid_Transformer/dataset_Autsl/date_19_02_2021_17.34.12'

```

Folder structure:
```

├── checkpoint_folder/
│   ├── model_epoch_X.pth/ - trained model weights
│   └── log/ - default logdir for logging output
```