import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, process_val_result, plot_comparison
import sys

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, train_criterion, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None, val_criterion=None):
        super().__init__(model, train_criterion, metrics, optimizer, config, val_criterion)
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
        self.valid_data_loader = valid_data_loader

        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
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
        total_metrics = np.zeros(len(self.metrics))
        

        with tqdm(self.data_loader) as progress:

            for batch_idx, (data, label) in enumerate(progress):
                
                progress.set_description_str(f'Train epoch {epoch}')
                
                data, label = data.to(self.device), label.to(self.device)
                
                output = self.model(data)

                # loss, loss_track = self.train_criterion(output, label)
                loss = self.train_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization

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
                total_metrics += self._eval_metrics(output, label)


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
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'learning rate': self.lr_scheduler.get_lr()
        }


        if self.do_validation:
            val_log, val_fig = self._valid_epoch(epoch)
            log.update(val_log)
        else:
            val_log, val_fig = None, None
        if self.do_test:
            test_log = self._test_epoch(epoch)
            log.update(test_log)
        # else: 
        #     test_meta = [0,0]


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log, val_fig


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        print("start validation")
        self.model.eval()

        self.val_result = [] # Clear cache
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)

                    self.val_result.append(output) # added
                    # loss,_ = self.val_criterion(output, label)
                    loss = self.val_criterion(output, label, self.config["scaling_factor"]) # Changed for only localization

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, epoch=epoch, mode = 'valid')
                    self.writer.add_scalar({'loss': loss.item()})
                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, label)
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        #mse_final_loss, fig = self._eval_val_metric()
        fig = self._eval_val_metric(data, label, output) # Changed for only localization
        log_output = {'val_loss': total_val_loss / len(self.valid_data_loader), 
                      'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()}

        return log_output, fig

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        # results = np.zeros((len(self.test_data_loader.dataset), self.config['num_classes']), dtype=np.float32)
        # tar_ = np.zeros((len(self.test_data_loader.dataset),), dtype=np.float32)
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, label) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    
                    # loss,_ = self.val_criterion(output, label)
                    loss = self.val_criterion(output, label, self.config["scaling_factor"]) # Changed for only localization

                    self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, epoch=epoch, mode = 'test')
                    self.writer.add_scalar({'loss': loss.item()})
                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, label)
                    #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                    # results[indexs.cpu().detach().numpy().tolist()] = output.cpu().detach().numpy().tolist()
                    # tar_[indexs.cpu().detach().numpy().tolist()] = label.cpu().detach().numpy().tolist()

        return {
            'test_loss': total_test_loss / len(self.test_data_loader),
            'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
        }#,[results,tar_]


    def _warmup_epoch(self, epoch):
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
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
                total_metrics += self._eval_metrics(output, label)


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
            'metrics': (total_metrics / self.len_epoch).tolist(),
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
