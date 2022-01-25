import comet_ml
import argparse
import collections
#import sys
#import requests
#import socket
import torch
#import mlflow
#import mlflow.pytorch
from data_loader.MicroscopyDataloader_est import MicroscopyDataLoader_est
from data_loader.MicroscopyDataloader import MicroscopyDataLoader
from torch.utils.data import DataLoader
import model.loss as module_loss
import model.metric_v2 as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer_main import *
from trainer.est_main import *
from collections import OrderedDict
import random
import numpy as np
#import pixiedust


def main(config: ConfigParser):
   
    # parameters for the training and testing set

    params_est = {'batch_size':config['est_dataset']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}
    params_valid = {'batch_size':config['validation_dataset']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}

    est_file_names = {'noise_image_name':config['est_dataset']['noise_image_name'],
    'GT_list_name':config['est_dataset']['GT_list_name'], 
'file_folder':config['est_dataset']['file_folder'],                                   
'batch_size':config['est_dataset']['batch_size'],
'setup_params':config['microscopy_params']['setup_params']}

    vallidation_file_names = {'noise_image_name':config['validation_dataset']['noise_image_name'],
'GT_image_name':config['validation_dataset']['GT_image_name'],         'GT_list_name':config['validation_dataset']['GT_list_name'], 
'file_folder':config['validation_dataset']['file_folder'],                                   
'batch_size':config['validation_dataset']['batch_size'],
'setup_params':config['microscopy_params']['setup_params']}
    

    list_ID_est = np.int_(np.arange(1,config['est_dataset']['number_images']+1))
    est_set = MicroscopyDataLoader_est(list_ID_est, **est_file_names)
    est_generator = DataLoader(est_set, **params_est)

    list_ID_validation = np.int_(np.arange(1,config['validation_dataset']['number_images']+1))
    validation_set = MicroscopyDataLoader(list_ID_validation, **vallidation_file_names)
    validation_generator = DataLoader(validation_set, **params_valid)
    

    # build model architecture, then print to console
    model = getattr(module_arch, config["arch"]["type"])()

    # get function handles of loss and metrics
    
    
    val_loss_metri = getattr(module_metric, config['val_loss'])
    

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)


    trainer = Est(model, optimizer,
                    config=config,
                    valid_data_loader=None,
                    est_data_loader=est_generator,
                    metric_for_val = val_loss_metri)
                                                                           

    #trainer.train()
    trainer.est()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training parameters')
    args.add_argument('-c', '--config', default="config_orientations_v2.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/home/wut/Documents/Deep-SMOLM/data/save/models/intensity_weighted_moments_training_sym_90/0112_220013/model_best.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')),
        CustomArgs(['--alpha', '--alpha'], type=float, target=('train_loss', 'args', 'alpha')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
    ]
    
    config = ConfigParser.get_instance(args, options)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache() 
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    main(config)

# %%