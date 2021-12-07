import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, process_val_result, plot_comparison, plot_zoom_in
import matplotlib.pyplot as plt
import sys

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, optimizer, config, data_loader,
                 valid_data_loader_1SM=None, valid_data_loader_2SMs=None, test_data_loader=None, lr_scheduler=None, len_epoch=None,metric_for_val_1SM = None,metric_for_val_2SMs = None,metric_for_test=None):
        super().__init__(model, train_criterion, optimizer, config, metric_for_val_1SM,metric_for_val_2SMs,metric_for_test)
        self.config = config
        if self.config["train_loss"] == "cross_entropy":
            self.ce = True
        else:
            self.ce = False
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader_1SM = valid_data_loader_1SM
        self.valid_data_loader_2SMs = valid_data_loader_2SMs
        self.test_data_loader = test_data_loader

        self.do_validation_1SM = self.valid_data_loader_1SM is not None
        self.do_validation_2SMs = self.valid_data_loader_2SMs is not None
        self.do_test = self.test_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_1SM_loss_list: List[float] = []
        self.val_2SMs_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        #Visdom visualization
        

    def _eval_metrics(self, output, label):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            # to_compare = torch.zeros(output.shape[0], 1, output.shape[2], output.shape[3]).to(output.device)
            # to_compare = torch.zeros(output.shape[0], 4, output.shape[2], output.shape[3]).to(output.device) # Changed for only localization
            # to_compare = torch.cat([output, to_compare], 1)
            acc_metrics[i] += metric(output, label, self.ce)
            self.writer.add_scalar({'{}'.format(metric.__name__): acc_metrics[i]})
        return acc_metrics
    
    # def _eval_val_metric(self):
    #     recovery = torch.cat(self.val_result, 0)
    #     MSE_M_final, fig = process_val_result(recovery)
    #     self.writer.add_scalar({'Evaluation metric': MSE_M_final})
    #     return MSE_M_final, fig

    def _eval_val_metric(self, data, label, output): # Added for only localization
        rawx = data[:,0,:,:]
        rawy = data[:,1,:,:]
        if self.ce:
            estimated_loc = torch.argmax(output, 1).float()
        else:
            estimated_loc = output[:,0,:,:]
        gt_loc = label[:,0,:,:] 
        total_num = output.shape[0]
        samp_to_show = int(torch.randint(0,total_num,[1]))
        fig = plot_comparison(rawx[samp_to_show], rawy[samp_to_show], gt_loc[samp_to_show], estimated_loc[samp_to_show])

        return fig

    def _eval_val_metric_zoom(self, data, label, output, n_sm = 1): # Added for only localization
        rawx = data[:,0,:,:]
        rawy = data[:,1,:,:]
        if self.ce:
            estimated_loc = torch.argmax(output, 1).float()
        else:
            estimated_loc = output[:,0,:,:]
        gt_loc = label[:,0,:,:] 
        total_num = output.shape[0]
        W = output.shape[-1]
        crop = 50

        samp_to_show = int(torch.randint(0,total_num,[1]))
        fig = plot_comparison(rawx[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)], 
                             rawy[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)], 
                             gt_loc[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)], 
                             estimated_loc[samp_to_show,int(W/2-crop):int(W/2+crop),int(W/2-crop):int(W/2+crop)])
        fig2 = plot_zoom_in(gt_loc[samp_to_show], estimated_loc[samp_to_show], n_sm = n_sm)

        return fig, fig2

    def _visualize_prediction(self, label, output): # Added for only localization
        self.gt_centered = torch.zeros(25,25)
        missing_pred = 0
        for i in range(output.shape[0]):
            location_pred = torch.argmax(output[i][0])
            gt_loc =  torch.argmax(label[i,0,:,:]) 
            location_pred_row = (location_pred / output[i][0].shape[1]).int()
            location_pred_col = (location_pred % output[i][0].shape[1]).int()
            location_gt_row = (gt_loc / output[i][0].shape[1]).int()
            location_gt_col = (gt_loc % output[i][0].shape[1]).int()
#             print("gt",location_gt_row, location_gt_col)
#             print("pred",location_pred_row, location_pred_col)
            shift_row = location_pred_row - location_gt_row
            shift_col = location_pred_col - location_gt_col
            #print(torch.max(output[i][0]))
            x_ind = int(self.gt_centered.shape[0]/2)+shift_row
            y_ind = int(self.gt_centered.shape[1]/2)+shift_col
            if x_ind < 25 and y_ind < 25 and x_ind >= 0 and y_ind >= 0:
                self.gt_centered[x_ind,y_ind] \
                                += torch.max(output[i][0]).item()
                #print(torch.max(label[i,0,:,:]))
            else:
                #print("can't find", torch.max(label[i,0,:,:]))
                warnings.warn("Some points predicted are outside the plot region!")
#                 self.gt_centered[0,0] \
#                                 += 5000
                missing_pred += 1
        fig, ax = plt.subplots()
        im = ax.imshow(self.gt_centered.cpu().numpy(), cmap="YlGnBu", origin="upper")
        fig.colorbar(im)
        ax.scatter(int(self.gt_centered.shape[0]/2),int(self.gt_centered.shape[1]/2), marker="x", color="black", s=100)
        ax.text(20,20, f"{missing_pred}/{output.shape[0]}")
        
        return fig

    def _visualize_prediction_2(self, target, output): # Added for only localization
        # Now add to distinguish different distance between SMs
        self.gt_centered = torch.zeros(9,25,25) # To differentiate differences
        missing_pred = torch.zeros(9).int()
        samp_dist = torch.zeros(9).int() # These two lines store how many samples each distance have
        target = target.cpu().numpy()
        output = output.cpu().numpy()
#         self.gt_centered = torch.zeros(25,25)
        center_x = int(self.gt_centered.shape[1]/2)
        center_y = int(self.gt_centered.shape[2]/2)
        for i in range(output.shape[0]):
            I_GT = target[i,0,:,:]
            x_GT, y_GT = find_gt(target[i,0,:,:])
            distance = ((x_GT[0] - x_GT[1]) ** 2 + (y_GT[0] - y_GT[1]) ** 2) ** (0.5)
            #print("distance", distance)
            # Check which distance range to belong
            if distance > 52 and distance < 56: # distance 9 (54)
                dist_ind = 8
            elif distance > 46: # distance 8(48)
                dist_ind = 7
            elif distance > 40: # distance 7(42)
                dist_ind = 6
            elif distance > 34: # distance 6(36)
                dist_ind = 5
            elif distance > 28: # distance 5(30)
                dist_ind = 4
            elif distance > 22: # distance 4(24)
                dist_ind = 3
            elif distance > 16: # distance 3(18)
                dist_ind = 2
            elif distance > 10: # distance 2(12)
                dist_ind = 1
            elif distance > 4: # distance 1(6)
                dist_ind = 0
            else:
                raise ValueError("Distance Wrong!")
            
            I_thresh = 0.3*np.min([I_GT[int(x_GT[0]), int(y_GT[0])], I_GT[int(x_GT[1]), int(y_GT[1])]])
            I_est_save,x_est_save,y_est_save = postprocessing_loc(output[i,0,:,:], 
                                                                  I_thresh, 
                                                                  [3,3], 
                                                                  matlab_style_gauss2D(shape=(7,7),sigma=1.0))
            n_pred_sm = np.size(x_est_save)
            #print("points predicted", n_pred_sm)
            if n_pred_sm != 2:
                missing_pred[dist_ind] += 1
            samp_dist[dist_ind] += 1
            
            for j in range(n_pred_sm):
                distance_1 = np.abs(x_est_save[j] - x_GT[0]) + np.abs(y_est_save[j] - y_GT[0])
                distance_2 = np.abs(x_est_save[j] - x_GT[1]) + np.abs(y_est_save[j] - y_GT[1])
                if distance_1 < distance_2:
                    shift_row = x_est_save[j] - x_GT[0]
                    shift_col = y_est_save[j] - y_GT[0]
                else:
                    shift_row = x_est_save[j] - x_GT[1]
                    shift_col = y_est_save[j] - y_GT[1]
                if center_x + shift_row >= self.gt_centered.shape[1] or center_y + shift_col >= self.gt_centered.shape[2] \
                        or center_x + shift_row < 0 or center_y + shift_col < 0:
                    warnings.warn("Some points predicted are outside the plot region!")
                    self.gt_centered[dist_ind,0,0] += 5000
                else:
                    self.gt_centered[dist_ind,center_x+shift_row,center_y+shift_col] += I_est_save[j].astype('float32')
        
        fig, ax = plt.subplots(3,3, figsize = (16,16))
        for i in range(9):
            im = ax[int(i/3), int(i%3)].imshow(self.gt_centered.cpu().numpy()[i], cmap="YlGnBu", origin="upper")
            plt.colorbar(im, ax = ax[int(i/3), int(i%3)])
            ax[int(i/3), int(i%3)].scatter(int(self.gt_centered.shape[1]/2),
                                           int(self.gt_centered.shape[2]/2), marker="x", color="black", s=100)
            ax[int(i/3), int(i%3)].text(20,20, f"{missing_pred[i]}/{samp_dist[i]}")
            ax[int(i/3), int(i%3)].set_title(f"Distance: {i+1}")

        return fig

    

    def _train_epoch(self, epoch):
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
                
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.to(self.device)
                
                output = self.model(data)

                # loss, loss_track = self.train_criterion(output, label)
                loss,loss_track = self.train_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization

                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)
                self.writer.add_scalar({'loss': loss.item()})
                # self.writer.add_scalar({'loss_theta': loss_track[0]})
                # self.writer.add_scalar({'loss_phi': loss_track[1]})
                # self.writer.add_scalar({'loss_gamma': loss_track[2]})
                # self.writer.add_scalar({'loss_intensity': loss_track[3]})
                
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                #total_metrics += self._eval_metrics(output, label)


                if batch_idx % self.log_step == 0:

                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
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
            val_log_1SM,fig_1SM = self._valid_epoch_1SM(epoch)
            log.update(val_log_1SM)
            
        else:
            val_log_1SM = None
            fig_1SM = None

        if self.do_validation_2SMs:
            val_log_2SMs,fig_2SMs = self._valid_epoch_2SMs(epoch)
            log.update(val_log_2SMs)
        else:
            val_log_2SMs = None
            fig_2SMs = None


        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)
        else:
            test_log = None


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        
        return log, fig_1SM, fig_2SMs


    def _valid_epoch_1SM(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        print("start validation_1SM")
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
            with tqdm(self.valid_data_loader_1SM) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    B,L,H,W = np.shape(data)
                    self.val_result.append(output) # added
                    
                    loss,loss_detail = self.metric_for_val_1SM(self.config,output, label) # Changed for only localization
                    TP_cur,TN_cur,FP_cur,MSE_loc_cur,MSE_I_cur,count_cur = loss_detail[0],loss_detail[1],loss_detail[2],loss_detail[3],loss_detail[4],loss_detail[5],
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader_1SM) + batch_idx, epoch=epoch, mode = 'valid')                   
                    self.writer.add_scalar({'Jaccard_1SM': loss[0]})
                    self.writer.add_scalar({'RMSE_loc_1SM': loss[1]})
                    self.writer.add_scalar({'RMSE_I_1SM': loss[2]})
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
        fig = self._eval_val_metric_zoom(data, label, output) # Changed for only localization
        jaccard = TP/(TN+TP+FP)
        if count==0:
            RMSE_loc = 1e10
            RMSE_I = 1e10
        else:
            RMSE_loc = (MSE_loc/count)**(1/2)
            RMSE_I = (MSE_I/count)**(1/2)
        log_output = {'Jaccard_1SM': jaccard,
                     'RMSE_loc_1SM':RMSE_loc,
                     'RMSE_I_1SM':RMSE_I}

        save_output = [jaccard,RMSE_loc,RMSE_I]
        return log_output,fig


    def _valid_epoch_2SMs(self, epoch):
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
                    
                    loss,loss_detail = self.metric_for_val_1SM(self.config,output, label) # Changed for only localization
                    TP_cur,TN_cur,FP_cur,MSE_loc_cur,MSE_I_cur,count_cur = loss_detail[0],loss_detail[1],loss_detail[2],loss_detail[3],loss_detail[4],loss_detail[5],
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader_1SM) + batch_idx, epoch=epoch, mode = 'valid')                   
                    self.writer.add_scalar({'Jaccard_2SMs': loss[0]})
                    self.writer.add_scalar({'RMSE_loc_2SMs': loss[1]})
                    self.writer.add_scalar({'RMSE_I_2SMs': loss[2]})
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
        fig = self._eval_val_metric_zoom(data, label, output) # Changed for only localization
        jaccard = TP/(TN+TP+FP)
        if count==0:
            RMSE_loc = 1e10
            RMSE_I = 1e10
        else:
            RMSE_loc = (MSE_loc/count)**(1/2)
            RMSE_I = (MSE_I/count)**(1/2)
        log_output = {'Jaccard_2SMs': jaccard,
                     'RMSE_loc_2SMs':RMSE_loc,
                     'RMSE_I_2SMs':RMSE_I}

        save_output = [jaccard,RMSE_loc,RMSE_I]
        return log_output,fig



    def _test_epoch(self, epoch):
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

                    self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, epoch=epoch, mode = 'test')
                    self.writer.add_scalar({'loss': loss.item()})
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                  
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                    # results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
                    # tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()
        loss = total_test_loss / len(self.test_data_loader)
        save_output = [loss]
        return {
            'test_loss': loss
        }


    def _warmup_epoch(self, epoch):
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

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.writer.add_scalar('loss', loss.item())
                self.train_loss_list.append(loss.item())
                total_loss += loss.item()


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

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
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.do_test:
            test_log, test_meta = self._test_epoch(epoch)
            log.update(test_log)
        else: 
            test_meta = [0,0]

        return log


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
