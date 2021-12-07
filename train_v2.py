import comet_ml
import argparse
import collections
import sys
import requests
import socket
import torch
#import mlflow
#import mlflow.pytorch
from data_loader.MicroscopyDataloader import MicroscopyDataLoader
from torch.utils.data import DataLoader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from collections import OrderedDict
import random
from IPython.core.debugger import set_trace
import numpy as np
#import pixiedust


#%%pixie_debugger
def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)


def main(config: ConfigParser):
   
    logger = config.get_logger('train')

    
    params_train = {'batch_size':config['data_loader']['args']['batch_size'], 'shuffle':config['data_loader']['args']['shuffle'],
'num_workers':config['data_loader']['args']['num_workers']}
    
    
    params_test = {'batch_size':config['data_loader']['args']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}
    
    train_test_file_names = {'noise_image_name':config['training_dataset']['noise_image_name'],
'GT_image_name':config['training_dataset']['GT_image_name'],         'GT_list_name':config['training_dataset']['GT_list_name'], 
'file_folder':config['training_dataset']['file_folder'],                                   
'batch_size':config['data_loader']['args']['batch_size'],
'setup_params':config['microscopy_params']['setup_params']}
    
    vallidation_file_names_1SM = {'noise_image_name':config['validation_dataset_1SM']['noise_image_name'],
'GT_image_name':config['validation_dataset_1SM']['GT_image_name'],         'GT_list_name':config['validation_dataset_1SM']['GT_list_name'], 
'file_folder':config['validation_dataset_1SM']['file_folder'],                                   
'batch_size':config['validation_dataset_1SM']['batch_size'],
'setup_params':config['microscopy_params']['setup_params']}

    vallidation_file_names_2SMs = {'noise_image_name':config['validation_dataset_2SMs']['noise_image_name'],
'GT_image_name':config['validation_dataset_2SMs']['GT_image_name'],         'GT_list_name':config['validation_dataset_2SMs']['GT_list_name'], 
'file_folder':config['validation_dataset_2SMs']['file_folder'],                                   
'batch_size':config['validation_dataset_2SMs']['batch_size'],
'setup_params':config['microscopy_params']['setup_params']}

    params_valid = {'batch_size':config['validation_dataset_2SMs']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}
    
      
   
    number_images = config['training_dataset']['number_images']  
    percentage = config['trainer']['percent']                                                                   
    numb_training = np.floor(number_images*percentage) 
    numb_testing = np.floor(number_images*(1-percentage))                                       
    # instantiate the data class and create a datalaoder for training
    list_ID_train = np.int_(np.arange(1,numb_training+1))
    training_set = MicroscopyDataLoader(list_ID_train, **train_test_file_names)
    training_generator = DataLoader(training_set, **params_train)
    

    
    list_ID_test = np.int_(np.arange(numb_training+1,numb_training+numb_testing+1))
    test_set = MicroscopyDataLoader(list_ID_test, **train_test_file_names)
    test_generator = DataLoader(test_set, **params_test)
    batch_size = config['data_loader']['args']['batch_size']
    print(len(training_generator)*batch_size, len(test_generator)*batch_size)

    # instantiate the data class and create a datalaoder for validation
    list_ID_validation_1SM = np.int_(np.arange(1,config['validation_dataset_1SM']['number_images']+1))
    validation_set_1SM = MicroscopyDataLoader(list_ID_validation_1SM, **vallidation_file_names_1SM)
    validation_generator_1SM = DataLoader(validation_set_1SM, **params_valid)
    

    list_ID_validation_2SMs = np.int_(np.arange(1,config['validation_dataset_2SMs']['number_images']+1))
    validation_set_2SMs = MicroscopyDataLoader(list_ID_validation_2SMs, **vallidation_file_names_2SMs)
    validation_generator_2SMs = DataLoader(validation_set_2SMs, **params_valid)
    



    # build model architecture, then print to console
    model = config.initialize('arch', module_arch, config["scaling_factor"])

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(training_generator.dataset, 'num_raw_example'):
        num_examp = training_generator.dataset.num_raw_example
    else:
        num_examp = len(training_generator.dataset)

    # train_loss = getattr(module_loss, config['train_loss']['type'])(num_examp=num_examp, num_classes=config['num_classes'],
    #                                                         alpha=config['train_loss']['args']['alpha'])
    train_loss = getattr(module_loss, config['train_loss'])
    
    val_1SM_loss = getattr(module_metric, config['val_1SM_loss'])
    val_2SMs_loss = getattr(module_metric, config['val_2SMs_loss'])
    test_loss = getattr(module_loss, config['test_loss'])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)


    trainer = Trainer(model, train_loss, optimizer,
                      config=config,
                      data_loader=training_generator,
                      valid_data_loader_1SM=validation_generator_1SM,
                      valid_data_loader_2SMs=validation_generator_2SMs,
                      test_data_loader=test_generator,
                      lr_scheduler=lr_scheduler,
                      metric_for_val_1SM = val_1SM_loss,
                      metric_for_val_2SMs = val_2SMs_loss,
                      metric_for_test=test_loss)
                                                                           

    trainer.train()
    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    cfg_trainer = config['trainer']


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config_orientations_v2.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
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
