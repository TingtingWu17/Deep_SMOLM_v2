from typing import TypeVar, List, Tuple
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
#from logger import TensorboardWriter
from logger import CometWriter
import numpy as np
from utils import process_val_result
import scipy.io as sio

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, train_criterion, optimizer, config, metric_for_val_1SM,metric_for_val_2SMs,metric_for_test):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)

        self.val_result = [] # Stack all validation result later

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.train_criterion = train_criterion #.to(self.device)
        
        
        self.metric_for_val_1SM = metric_for_val_1SM
        self.metric_for_val_2SMs = metric_for_val_2SMs
        self.metric_for_test = metric_for_test

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
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

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        # self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer = CometWriter(
            self.logger,
            project_name = "deep-smolm",
            experiment_name = config['exper_name'],
            api_key = config['comet']['api'],
            log_dir = config.log_dir,
            offline = config['comet']['offline'])

        self.writer.log_hyperparams(config.config)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epochs number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            if epoch <= self.config['trainer']['warmup']:
                result = self._warmup_epoch(epoch)
            else:
                result, fig_1SM, fig_angles_1SM, fig_2SMs= self._train_epoch(epoch)

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

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            if (epoch - 1) % 1 == 0:
                figure_name = f"Epoch {epoch} validation result 1SM"
                self.writer.add_plot(figure_name, fig_1SM)
                figure_name = f"Epoch {epoch} validation result 1SM angles"
                self.writer.add_plot(figure_name, fig_angles_1SM)
            if (epoch - 1) % 1 == 0:
                figure_name = f"Epoch {epoch} validation result 2SMs"
                self.writer.add_plot(figure_name, fig_2SMs)
        sio.savemat("save_output.mat",{'train_loss':train_loss,'Jaccard_1SM':Jaccard_1SM,'Jaccard_2SMs':Jaccard_2SMs,'RMSE_I_1SM':RMSE_I_1SM,'RMSE_I_2SMs':RMSE_I_2SMs,
        'RMSE_loc_1SM':RMSE_loc_1SM,'RMSE_loc_2SMs':RMSE_loc_2SMs,'test_loss':test_loss})
                        
                
                

    
    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        # self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth at: {} ...".format(best_path))


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


