import comet_ml
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.MicroscopyDataloader import MicroscopyDataLoader
from torch.utils.data import DataLoader
import numpy as np
from test_utils import Tester
from utils2 import writeInfor2comet


def main(config):
    logger = config.get_logger('test')

    params_test = {'batch_size':config['data_loader']['args']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}

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

    list_ID_validation_1SM = np.int_(np.arange(1,config['validation_dataset_1SM']['number_images']+1))
    validation_set_1SM = MicroscopyDataLoader(list_ID_validation_1SM, **vallidation_file_names_1SM)
    validation_generator_1SM = DataLoader(validation_set_1SM, **params_valid)
    

    list_ID_validation_2SMs = np.int_(np.arange(1,config['validation_dataset_2SMs']['number_images']+1))
    validation_set_2SMs = MicroscopyDataLoader(list_ID_validation_2SMs, **vallidation_file_names_2SMs)
    validation_generator_2SMs = DataLoader(validation_set_2SMs, **params_valid)

    path_results = config['CNN_best_results']


    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)
    checkpoint = torch.load(path_results)
    model.load_state_dict(checkpoint['state_dict'])

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['test_loss'])
    metric_for_val_1SM = getattr(module_metric,config['val_1SM_loss'])
    metric_for_val_2SMs = getattr(module_metric,config['val_2SMs_loss'])


    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    #total_metrics = torch.zeros(len(metric_fns))
    total_metrics = 0.0


    #build the model and data infor into structure para
    tester = Tester(model,config,
                    valid_data_loader_1SM=validation_generator_1SM,                
                    metric_for_val_1SM=metric_for_val_1SM,                     
                    valid_data_loader_2SMs=validation_generator_2SMs,                   
                    metric_for_val_2SMs=metric_for_val_2SMs)
    writeInfor2comet(tester)
    tester._valid_epoch_1SM(1)




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default="config_test.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.get_instance(args, '')
    #config = ConfigParser(args)
    main(config)
