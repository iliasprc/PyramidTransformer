import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.multimodal.mutimodal_model import RGBD_Transformer
from models.vmz.eca_ir_csn_152 import eca_ir_csn_152
from models.vmz.ir_csn_152 import ir_csn_152
from models.vmz.pyramid_transformer import ir_csn_152_transformer


def SLR_video_encoder(config,N_classes):


    if config.model.name == 'IR_CSN_152':
        return ir_csn_152(pretraining="ig65m_32frms", pretrained=True, progress=False, num_classes=N_classes)
    elif config.model.name == 'ECA_IR_CSN_152':
        return eca_ir_csn_152(pretraining="ig65m_32frms", pretrained=True, progress=False, num_classes=N_classes)

    elif config.model.name == 'Pyramid_Transformer':
        return ir_csn_152_transformer(pretraining="ig65m_32frms", pretrained=True, progress=False, num_classes=N_classes)


def RGBD_model(config, N_classes):
    if config.model.name == 'RGBD_Transformer':
        m = RGBD_Transformer(config, N_classes)
        return m


def showgradients(model):
    for name, param in model.named_parameters():
        print(name, ' ', type(param.data), param.size())
        print("GRADS= \n", param.grad)


def save_checkpoint_slr(model, optimizer, epoch, wer, checkpoint, name, save_seperate_layers=False, is_best=False):
    state = {}
    if (save_seperate_layers):
        for name1, module in model.named_children():
            # print(name1)
            state[name1 + '_dict'] = module.state_dict()

    state['model_dict'] = model.state_dict()
    state['optimizer_dict'] = optimizer.state_dict()
    state['wer'] = str(wer)
    state['epoch'] = str(epoch)
    filepath = os.path.join(checkpoint, name + '.pth')

    if is_best:
        filepath = os.path.join(checkpoint, 'best' + name + f'.pth')

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath, _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint, model, strict=True, optimizer=None, load_seperate_layers=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint1 = torch.load(checkpoint, map_location='cpu' )
    pretrained_dict = checkpoint1['model_dict']
    model_dict = model.state_dict()
    print(pretrained_dict.keys() )
    print(model_dict.keys())
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dictnew = {}
    for k, v in pretrained_dict.items():
        if ' model' in k:
            k = k[6:]
        pretrained_dictnew[k] = v
    # # for k, v in pretrained_dict.items():
    # #     k = k.strip('model.')
    # #     pretrained_dictnew[k] = v
    print(pretrained_dictnew.keys())
    # #pretrained_dict = {k: v for k, v in pretrained_dictnew.items() if k in model_dict}
    #
    # # # 2. overwrite entries in the existing state dict
    # # model_dict.update(pretrained_dict)
    # # # 3. load the new state dict
    # # #model.load_state_dict(pretrained_dict)

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))


    if (not load_seperate_layers):

        model.load_state_dict(pretrained_dictnew , strict=strict)


    epoch = 0
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return checkpoint, epoch


def load_checkpoint_modules(checkpoint, model, strict=True, optimizer=None, load_seperate_layers=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location='cpu')
    print(checkpoint.keys())
    if (not load_seperate_layers):

        model.load_state_dict(checkpoint['model_dict'], strict=strict)
    else:
        for name1, module in model.named_children():
            # print(name1)
            module.load_state_dict(checkpoint[name1 + '_dict'], strict=strict)

    epoch = 0
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    return checkpoint, epoch




def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(classname)
        m.eval()
    return m


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_weights_linear(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    return m


def _initialize_weights1d(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def init_weights_rnn(model):
    for m in model.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    return model


def weights_init_uniform(net):
    for name, param in net.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.uniform_(param, a=-0.1, b=0.1)
    return net


def select_optimizer(model, config, checkpoint=None):
    opt = config['optimizer']['type']
    lr = config['optimizer']['lr']
    if (opt == 'Adam'):
        print(" use optimizer Adam lr ", lr)
        optimizer = optim.Adam(model.parameters(), lr=float(config['optimizer']['lr']), weight_decay=0.00001)
    elif (opt == 'SGD'):
        print(" use optimizer SGD lr ", lr)
        optimizer = optim.SGD(model.parameters(), lr=float(config['optimizer']['lr']), momentum=0.9)
    elif (opt == 'RMSprop'):
        print(" use RMS  lr", lr)
        optimizer = optim.RMSprop(model.parameters(), lr=float(config['optimizer']['lr']))
    if (checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
    if config['scheduler']['type'] == 'ReduceLRonPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=config['scheduler']['scheduler_factor'],
                                      patience=config['scheduler']['scheduler_patience'],
                                      min_lr=config['scheduler']['scheduler_min_lr'],
                                      verbose=config['scheduler']['scheduler_verbose'])
        return optimizer, scheduler

    return optimizer, None
