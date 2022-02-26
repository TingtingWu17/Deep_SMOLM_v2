import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
#from base import BaseTrainer
from trainer.val_plot_util import process_val_result, plot_comparison_v2, plot_angle_scatters, eval_val_metric_zoom
import matplotlib.pyplot as plt
import sys
from numpy import inf
from trainer.trainer_utils import *
from logger import CometWriter
import scipy.io as sio



def train_epoch(self, epoch):
    """
    Training logic for an epoch

    :param epoch: Current training epoch.
    :return: A log that contains all information you want to save.

    Note:
        If you have additional information to record, for example:
            > additional_log = {"x": x, "y": y}
        merge it with log before return. i.e.
            > log = {**log, **additional_log}
            > return log

        The metrics in log must have the key 'metrics'.
    """  
        
    self.model.train()

    total_loss = 0

    

    with tqdm(self.data_loader) as progress:

        for batch_idx, (data, label) in enumerate(progress):
           
           if batch_idx<=100000:
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.to(self.device)
                
                output = self.model(data)

                # loss, loss_track = self.train_criterion(output, label)
                loss,loss_track = self.train_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization

                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()

                ifSaveData = self.config["comet"]["savedata"]
                if ifSaveData == True:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)
                    self.writer.add_scalar({'loss': loss.item()})
                    self.writer.add_scalar({'loss_MSE': loss_track[0]})
                    self.writer.add_scalar({'loss_I': loss_track[1]})
                    self.writer.add_scalar({'loss_XX': loss_track[2]})
                    self.writer.add_scalar({'loss_YY': loss_track[3]})
                    self.writer.add_scalar({'loss_ZZ': loss_track[4]})
                    self.writer.add_scalar({'loss_XY': loss_track[5]})
                    self.writer.add_scalar({'loss_XZ': loss_track[6]})
                    self.writer.add_scalar({'loss_YZ': loss_track[7]})
                # self.writer.add_scalar({'loss_phi': loss_track[1]})
                # self.writer.add_scalar({'loss_gamma': loss_track[2]})
                # self.writer.add_scalar({'loss_intensity': loss_track[3]})
                
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                #total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:

                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        progress_bar(self,batch_idx),
                        loss.item()))
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break


                
        
    # if hasattr(self.data_loader, 'run'):
    #     self.data_loader.run()

    log = {
        'loss': total_loss / self.len_epoch,
        'learning rate': self.lr_scheduler.get_lr()
    }
    

    if self.do_validation_1SM:
        val_log_1SM,fig_1SM,fig_angles_1SM = valid_epoch_1SM(self, epoch)
        log.update(val_log_1SM)
        
    else:
        val_log_1SM = None
        fig_1SM = None
        fig_angles_1SM = None

    if self.do_validation_2SMs:
        val_log_2SMs,fig_2SMs = valid_epoch_2SMs(self, epoch)
        log.update(val_log_2SMs)
    else:
        val_log_2SMs = None
        fig_2SMs = None


    if self.do_test:
        test_log = test_epoch(self, epoch)
        log.update(test_log)
    else:
        test_log = None


    if self.lr_scheduler is not None:
        self.lr_scheduler.step()

    
    return log, fig_1SM, fig_angles_1SM, fig_2SMs


def valid_epoch_1SM(self, epoch):
    """
    Validate after training an epoch

    :return: A log that contains information about validation

    Note:
        The validation metrics in log must have the key 'val_metrics'.
    """
    print("start validation_1SM")
    TP = 0
    FN = 0
    FP = 0
    bias_con_loc = 0
    MSE_loc = 0
    MSE_I = 0
    count =0
    has_SM1=0

    orien_est_all=[]
    orient_GT_all = []
    M_est_all = []
    bias_con_loc_x = 0
    bias_con_loc_y = 0
    std_con_loc_x = 0
    std_con_loc_y = 0

    self.model.eval()

    self.val_result = [] # Clear cache
    total_val_loss = 0
    
    with torch.no_grad():
        with tqdm(self.valid_data_loader_1SM) as progress:
            
            for batch_idx, (data, label) in enumerate(progress):
                progress.set_description_str(f'Valid epoch {epoch}')
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                B,L,H,W = np.shape(data)
                self.val_result.append(output) # added
                
                loss,loss_detail,bias_con_related,orien_est,orienta_GT,M_est,I_GT,I_est = self.metric_for_val_1SM(self.config,output, label) # Changed for only localization
               
                TP_cur,FN_cur,FP_cur,MSE_loc_cur,MSE_I_cur,count_cur = loss_detail[0],loss_detail[1],loss_detail[2],loss_detail[3],loss_detail[4],loss_detail[5]
                
                #self.val_1SM_loss_list.append(loss.item())
                TP += TP_cur
                FN += FN_cur
                FP += FP_cur
                MSE_loc += MSE_loc_cur
                MSE_I += MSE_I_cur
                count +=count_cur
             

                if np.size(orien_est)>0:                 
                    if has_SM1==0:
                        orien_est_all = orien_est
                        orient_GT_all = orienta_GT
                        M_est_all = M_est
                        bias_con_all = bias_con_related
                        has_SM1=1
                    else:
                        orien_est_all =  np.concatenate((orien_est_all,orien_est),axis=0)
                        orient_GT_all =  np.concatenate((orient_GT_all,orienta_GT),axis=0)
                        M_est_all = np.concatenate((M_est_all,M_est),axis=1)
                        bias_con_all =  np.concatenate((bias_con_all,bias_con_related),axis=0)

                
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    # add histogram of model parameters to the tensorboard
    # for name, p in self.model.named_parameters():
    #     self.writer.add_histogram(name, p, bins='auto')



    #mse_final_loss, fig = self._eval_val_metric()
    fig = eval_val_metric_zoom(data, label, output) # Changed for only localization
    fig_angles = plot_angle_scatters(orien_est_all,orient_GT_all)
    jaccard = TP/(FN+TP+FP)
    if count>0:
        bias_con_loc_x = np.mean(bias_con_all[:,0])
        bias_con_loc_y = np.mean(bias_con_all[:,1])
        std_con_loc_x = np.std(bias_con_all[:,0])
        std_con_loc_y = np.std(bias_con_all[:,1])
        

    if count==0:
        RMSE_loc = 0
        RMSE_I = 0
    else:
        RMSE_loc = (MSE_loc/count)**(1/2)
        RMSE_I = (MSE_I/count)**(1/2)
    log_output = {'Jaccard_1SM': jaccard,
                    'RMSE_loc_1SM':RMSE_loc,
                    'RMSE_I_1SM':RMSE_I}

    ifSaveData = self.config["comet"]["savedata"]
    if ifSaveData == True:
        self.writer.set_step(epoch, epoch=epoch, mode = '1SM_valid')                   
        self.writer.add_scalar({'Jaccard_1SM': jaccard})
        self.writer.add_scalar({'RMSE_loc_1SM': RMSE_loc})
        self.writer.add_scalar({'RMSE_I_1SM': RMSE_I})
        self.writer.add_scalar({'bias x': bias_con_loc_x})
        self.writer.add_scalar({'bias y': bias_con_loc_y})
        self.writer.add_scalar({'std x': std_con_loc_x})
        self.writer.add_scalar({'std y': std_con_loc_y})

    save_output = [jaccard,RMSE_loc,RMSE_I]
    return log_output,fig,fig_angles


def valid_epoch_2SMs(self, epoch):
    """
    Validate after training an epoch

    :return: A log that contains information about validation

    Note:
        The validation metrics in log must have the key 'val_metrics'.
    """
    print("start validation_2SMs")
    TP = 0
    TN = 0
    FP = 0
    MSE_loc = 0
    MSE_I = 0
    count =0

    self.model.eval()

    self.val_result = [] # Clear cache
    total_val_loss = 0
    
    with torch.no_grad():
        with tqdm(self.valid_data_loader_2SMs) as progress:
            for batch_idx, (data, label) in enumerate(progress):
                progress.set_description_str(f'Valid epoch {epoch}')
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                B,L,H,W = np.shape(data)
                self.val_result.append(output) # added
                
                loss,loss_detail,bias_con_related,orien_est,orienta_GT,M_est,I_GT,I_est = self.metric_for_val_1SM(self.config,output, label) # Changed for only localization
                TP_cur,TN_cur,FP_cur,MSE_loc_cur,MSE_I_cur,count_cur = loss_detail[0],loss_detail[1],loss_detail[2],loss_detail[3],loss_detail[4],loss_detail[5],
                ifSaveData = self.config["comet"]["savedata"]

                #self.val_1SM_loss_list.append(loss.item())
                TP += TP_cur
                TN += TN_cur
                FP += FP_cur
                MSE_loc += MSE_loc_cur
                MSE_I += MSE_I_cur
                count +=count_cur

                
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    # add histogram of model parameters to the tensorboard
    # for name, p in self.model.named_parameters():
    #     self.writer.add_histogram(name, p, bins='auto')

    #mse_final_loss, fig = self._eval_val_metric()
    fig = eval_val_metric_zoom(data, label, output) # Changed for only localization
    jaccard = TP/(TN+TP+FP)
    if count==0:
        RMSE_loc = 0
        RMSE_I = 0
    else:
        RMSE_loc = (MSE_loc/count)**(1/2)
        RMSE_I = (MSE_I/count)**(1/2)
    log_output = {'Jaccard_2SMs': jaccard,
                    'RMSE_loc_2SMs':RMSE_loc,
                    'RMSE_I_2SMs':RMSE_I}

    save_output = [jaccard,RMSE_loc,RMSE_I]

    ifSaveData = self.config["comet"]["savedata"]
    if ifSaveData == True:
        self.writer.set_step(epoch, epoch=epoch, mode = '2SM_valid')                   
        self.writer.add_scalar({'Jaccard_1SM': jaccard})
        self.writer.add_scalar({'RMSE_loc_1SM': RMSE_loc})
        self.writer.add_scalar({'RMSE_I_1SM': RMSE_I})

    return log_output,fig



def test_epoch(self, epoch):
    """
    Test after training an epoch

    :return: A log that contains information about test

    Note:
        The Test metrics in log must have the key 'val_metrics'.
    """
    self.model.eval()
    total_test_loss = 0

    # results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
    # tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
    with torch.no_grad():
        with tqdm(self.test_data_loader) as progress:
            for batch_idx, (data, label) in enumerate(progress):
                progress.set_description_str(f'Test epoch {epoch}')
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                
                # loss,_ = self.val_criterion(output, label)
                loss,loss_track = self.metric_for_test(output, label, self.config["scaling_factor"]) # Changed for only localization

                ifSaveData = self.config["comet"]["savedata"]


                self.test_loss_list.append(loss.item())
                total_test_loss += loss.item()
                
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
                # tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()
    loss = total_test_loss / len(self.test_data_loader)
    save_output = [loss]

    if ifSaveData == True:
        self.writer.set_step(epoch, epoch=epoch, mode = 'test')
        self.writer.add_scalar({'loss': loss})

    return {
        'test_loss': loss
    }


def warmup_epoch(self, epoch):
    total_loss = 0
    self.model.train()

    data_loader = self.data_loader#self.loader.run('warmup')


    with tqdm(data_loader) as progress:
        for batch_idx, (data, label, _, indexs , _) in enumerate(progress):
            progress.set_description_str(f'Warm up epoch {epoch}')

            data, label = data.to(self.device), label.long().to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            out_prob = torch.nn.functional.softmax(output).data.detach()

            self.train_criterion.update_hist(indexs.cpu().detach().numpy().tolist(), out_prob)

            loss = torch.nn.functional.cross_entropy(output, label)

            loss.backward() 
            self.optimizer.step()

            #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            #self.writer.add_scalar('loss', loss.item())
            self.train_loss_list.append(loss.item())
            total_loss += loss.item()


            if batch_idx % self.log_step == 0:
                progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                    progress_bar(self,batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
    if hasattr(self.data_loader, 'run'):
        self.data_loader.run()
    log = {
        'loss': total_loss / self.len_epoch,
        'noise detection rate' : 0.0,
        'learning rate': self.lr_scheduler.get_lr()
    }

    if self.do_validation:
        val_log = valid_epoch(self,epoch)
        log.update(val_log)
    if self.do_test:
        test_log, test_meta = test_epoch(self,epoch)
        log.update(test_log)
    else: 
        test_meta = [0,0]

    return log



