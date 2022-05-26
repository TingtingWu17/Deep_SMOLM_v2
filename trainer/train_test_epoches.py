import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from numpy import inf
from trainer.trainer_utils import *
from model.metric_v2 import postprocessingv2, RMSE_1SM_resnet




def train_epoch(self, epoch):
    print(epoch)
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


    #with tqdm(self.data_loader) as progress:
    for batch_idx, (data, label,indx) in enumerate(self.data_loader):
        
        #print(batch_idx)
        #progress.set_description_str(f'Train epoch {epoch}')
        
        data, label = data.to(self.device), label.to(self.device)
        
        output = self.model(data)

        
        # loss, loss_track = self.train_criterion(output, label)
        loss,loss_track = self.train_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization

        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        #est = postprocessingv2(self.config, output, 1000*label[:,6:12,:,:], indx)

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


        # if batch_idx % self.log_step == 0:

            # progress.set_postfix_str(' {} Loss: {:.6f}'.format(
            #     progress_bar(self,batch_idx),
            #     loss.item()))
            #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if batch_idx == self.len_epoch:
            break


                    
            
        # if hasattr(self.data_loader, 'run'):
    #     self.data_loader.run()

    log = {
        'loss': total_loss / self.len_epoch,
        'learning rate': self.lr_scheduler.get_last_lr()
    }
    
    if self.do_test:
        test_log = test_epoch(self, epoch)
        log.update(test_log)
    else:
        test_log = None


    if self.lr_scheduler is not None:
        self.lr_scheduler.step()

    
    return log



def test_epoch(self, epoch):
    print(epoch)
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
        #with tqdm(self.test_data_loader) as progress:
        for batch_idx, (data, label,idx) in enumerate(self.test_data_loader):
            #print(batch_idx)
            #progress.set_description_str(f'Test epoch {epoch}')
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


# def warmup_epoch(self, epoch):
#     total_loss = 0
#     self.model.train()

#     data_loader = self.data_loader#self.loader.run('warmup')


#     with tqdm(data_loader) as progress:
#         for batch_idx, (data, label, _, indexs , _) in enumerate(progress):
#             progress.set_description_str(f'Warm up epoch {epoch}')

#             data, label = data.to(self.device), label.long().to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(data)
#             out_prob = torch.nn.functional.softmax(output).data.detach()

#             self.train_criterion.update_hist(indexs.cpu().detach().numpy().tolist(), out_prob)

#             loss = torch.nn.functional.cross_entropy(output, label)

#             loss.backward() 
#             self.optimizer.step()

#             #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
#             #self.writer.add_scalar('loss', loss.item())
#             self.train_loss_list.append(loss.item())
#             total_loss += loss.item()


#             if batch_idx % self.log_step == 0:
#                 progress.set_postfix_str(' {} Loss: {:.6f}'.format(
#                     progress_bar(self,batch_idx),
#                     loss.item()))
#                 #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#             if batch_idx == self.len_epoch:
#                 break
#     if hasattr(self.data_loader, 'run'):
#         self.data_loader.run()
#     log = {
#         'loss': total_loss / self.len_epoch,
#         'noise detection rate' : 0.0,
#         'learning rate': self.lr_scheduler.get_lr()
#     }

#     if self.do_validation:
#         val_log = valid_epoch(self,epoch)
#         log.update(val_log)
#     if self.do_test:
#         test_log, test_meta = test_epoch(self,epoch)
#         log.update(test_log)
#     else: 
#         test_meta = [0,0]

#     return log



