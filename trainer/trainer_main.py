import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
#from base import BaseTrainer
from utils import inf_loop
import matplotlib.pyplot as plt
import sys
from numpy import inf
from trainer.trainer_utils import *
from logger import CometWriter
import scipy.io as sio
#from train_test_epoches import *
from trainer.train_test_epoches import *

class Trainer:
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, optimizer, config, data_loader,
                 valid_data_loader_1SM=None, valid_data_loader_2SMs=None, test_data_loader=None, lr_scheduler=None, len_epoch=None,metric_for_val_1SM = None,metric_for_val_2SMs = None,metric_for_test=None):
        # pass all variables to self
        #prepare the training device
        # setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(self,config['n_gpu'])
        self.model = model.to(self.device)
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.config = config
        #self.val_result = [] # Stack all validation result later

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.train_criterion = train_criterion 
        self.optimizer = optimizer
        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = self.config.save_dir
        if self.config.resume is not None:
            resume_checkpoint(self,self.config.resume)

        #training metric
        self.metric_for_val_1SM = metric_for_val_1SM
        self.metric_for_val_2SMs = metric_for_val_2SMs
        self.metric_for_test = metric_for_test

        self.valid_data_loader_1SM = valid_data_loader_1SM
        self.valid_data_loader_2SMs = valid_data_loader_2SMs
        self.test_data_loader = test_data_loader

        self.do_validation_1SM = self.valid_data_loader_1SM is not None
        self.do_validation_2SMs = self.valid_data_loader_2SMs is not None
        self.do_test = self.test_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        #data save space 
        self.train_loss_list: List[float] = []
        self.val_1SM_loss_list: List[float] = []
        self.val_2SMs_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        self.val_result = []


        

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        ifSaveData = self.config["comet"]["savedata"]
        if ifSaveData == True:
            savedata2comet(self,epoch=0)

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            if epoch <= self.config['trainer']['warmup']:
                result = self._warmup_epoch(epoch)
            else:
                result= train_epoch(self,epoch)
            if self.do_validation_1SM:
                if epoch==1:
                    train_loss = result["loss"]
                    Jaccard_1SM = result["Jaccard_1SM"]
                    Jaccard_2SMs = result["Jaccard_2SMs"]
                    RMSE_I_1SM = result["RMSE_I_1SM"]
                    RMSE_I_2SMs = result["RMSE_I_2SMs"]
                    RMSE_loc_1SM = result["RMSE_loc_1SM"]
                    RMSE_loc_2SMs = result["RMSE_loc_2SMs"]
                    test_loss = result["test_loss"]

                else:
                    train_loss = np.append(train_loss,result["loss"])
                    Jaccard_1SM = np.append(Jaccard_1SM,result["Jaccard_1SM"])
                    Jaccard_2SMs = np.append(Jaccard_2SMs,result["Jaccard_2SMs"])
                    RMSE_I_1SM = np.append(RMSE_I_1SM,result["RMSE_I_1SM"])
                    RMSE_I_2SMs = np.append(RMSE_I_2SMs,result["RMSE_I_2SMs"])
                    RMSE_loc_1SM = np.append(RMSE_loc_1SM,result["RMSE_loc_1SM"])
                    RMSE_loc_2SMs = np.append(RMSE_loc_2SMs,result["RMSE_loc_2SMs"])
                    test_loss = np.append(test_loss,result["test_loss"])    
            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                log.update({key:value})
            
            
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            if ifSaveData == True:               
                savedata2comet(self,epoch,best=best)
                if self.do_validation_1SM:
                    sio.savemat("save_output.mat",{'train_loss':train_loss,'Jaccard_1SM':Jaccard_1SM,'Jaccard_2SMs':Jaccard_2SMs,'RMSE_I_1SM':RMSE_I_1SM,'RMSE_I_2SMs':RMSE_I_2SMs,
            'RMSE_loc_1SM':RMSE_loc_1SM,'RMSE_loc_2SMs':RMSE_loc_2SMs,'test_loss':test_loss})
            plt.close('all')





    


   



    