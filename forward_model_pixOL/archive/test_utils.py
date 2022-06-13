from logger import CometWriter
import torch
from tqdm import tqdm
import torch
from utils2 import _prepare_device,_eval_val_metric_zoom,plot_angle_scatters,plot_I_scatters
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



class Tester:
    def __init__(self, model, config, valid_data_loader_1SM,metric_for_val_1SM,valid_data_loader_2SMs,metric_for_val_2SMs):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = _prepare_device(self,config['n_gpu'])
        self.model = model.to(self.device)

        self.val_result = [] # Stack all validation result later

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        
        
        self.metric_for_val_1SM = metric_for_val_1SM
        self.metric_for_val_2SMs = metric_for_val_2SMs
        self.valid_data_loader_1SM = valid_data_loader_1SM
        self.valid_data_loader_2SMs = valid_data_loader_2SMs



    def _valid_epoch_1SM(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        print("start validation_1SM")
  

        self.model.eval()

        self.val_result = [] # Clear cache
        total_val_loss = 0

        images_per_state = self.config['validation_dataset_1SM']['images_per_state']
        epoches_per_state = images_per_state/self.config['validation_dataset_1SM']['batch_size']
        
        with torch.no_grad():
            with tqdm(self.valid_data_loader_1SM) as progress:
                
                for batch_idx, (data, label) in enumerate(progress):

                    if (batch_idx)%epoches_per_state==0:
                        TP = 0
                        TN = 0
                        FP = 0
                        MSE_loc = 0
                        MSE_I = 0
                        count =0
                        has_SM1=0
                        con_bias_x = 0
                        con_bias_y = 0
                        I_bias = 0
                        orien_est_all=[]
                        orient_GT_all = []
                        M_est_all = []
                        I_est_all = []
                        I_GT_all = []
                        index_all = []
                    
                    #%%%%%%%
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, label = data.to(self.device), label.to(self.device)
                    output = self.model(data)
                    B,L,H,W = np.shape(data)
                    #self.val_result.append(output) # added
                    
                    loss_detail,bias_con_related,orien_est,orienta_GT,M_est,I_est,I_GT,index_cur = self.metric_for_val_1SM(self.config,output, label) # Changed for only localization
                    TP_cur,TN_cur,FP_cur,count_cur = loss_detail[0],loss_detail[1],loss_detail[2],loss_detail[3]                  
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader_1SM) + batch_idx, epoch=epoch, mode = 'valid')   
                    #self.val_1SM_loss_list.append(loss.item())
                    TP += TP_cur
                    TN += TN_cur
                    FP += FP_cur
                    count +=count_cur


                    if np.size(orien_est)>0:                 
                        if has_SM1==0:
                            orien_est_all = orien_est
                            orient_GT_all = orienta_GT
                            M_est_all = M_est
                            has_SM1=1
                            I_est_all = I_est
                            I_GT_all = I_GT

                            con_bias_x =bias_con_related[0]
                            con_bias_y = bias_con_related[1]
                            I_bias = bias_con_related[2]
                            index_all = index_cur

                        else:
                            orien_est_all =  np.concatenate((orien_est_all,orien_est),axis=0)
                            orient_GT_all =  np.concatenate((orient_GT_all,orienta_GT),axis=0)
                            M_est_all = np.concatenate((M_est_all,M_est),axis=1)
                            I_est_all = np.concatenate((I_est_all,I_est),axis=0)
                            I_GT_all = np.concatenate((I_GT_all,I_GT),axis=0)
                            con_bias_x = np.concatenate((con_bias_x,bias_con_related[0]),axis=0)
                            con_bias_y = np.concatenate((con_bias_y,bias_con_related[1]),axis=0)
                            I_bias = np.concatenate((I_bias,bias_con_related[2]),axis=0)
                            index_all = np.concatenate((index_all,index_cur),axis=0)
                    
                    if (batch_idx+1)%epoches_per_state==0:
                        #mse_final_loss, fig = self._eval_val_metric()
                        fig = _eval_val_metric_zoom(self,data, label, output) # Changed for only localization
                        fig_angles = plot_angle_scatters(orien_est_all,orient_GT_all)
                        fig_I = plot_I_scatters(I_est_all,I_GT_all)
                        temp = np.concatenate((index_all,index_all,index_all),1)
                        fig_angles_thred = plot_angle_scatters(np.reshape(orien_est_all[temp],(TP,3)),np.reshape(orient_GT_all[temp],(TP,3)))
                        fig_I_thred = plot_I_scatters(I_est_all[index_all],I_GT_all[index_all])
                        jaccard = TP/(TN+TP+FP)

                        if count==0:
                            con_bias_x_all =1e10
                            con_bias_y_all = 1e10
                            std_loc = 1e10                     
                            con_bias_x_all2 = 1e10
                            con_bias_y_all2 = 1e10
                            I_bias_all = 1e10
                            std_I = 1e10
                            RMSE_theta = 1e10
                            RMSE_phi = 1e10
                            RMSE_gamma = 1e10
                            RMSE_phi2 = 1e10

                            
                            #evaluate within the Jaccard circle
                            std_loc_thred = 1e10
                            std_I_thred = 1e10
                            con_bias_x_all_thred = 1e10
                            con_bias_y_all_thred = 1e10
                            I_bias_all_thred = 1e10

                        else:
                            con_bias_x_all =np.mean(con_bias_x)
                            con_bias_y_all = np.mean(con_bias_y)
                            std_loc = np.mean(((con_bias_x-con_bias_x_all)**2+(con_bias_y-con_bias_y)**2))**(1/2)                        
                            con_bias_x_all2 =np.mean(con_bias_x[index_all])
                            con_bias_y_all2 = np.mean(con_bias_y[index_all])
                            I_bias_all = np.mean(I_bias)
                            std_I = np.mean(((I_bias-I_bias_all)**2))**(1/2)
                            RMSE_theta = np.mean((orien_est_all[:,0]-orient_GT_all[:,0]) ** 2)**(1/2)
                            RMSE_phi = np.mean((orien_est_all[:,1]-orient_GT_all[:,1]) ** 2)**(1/2)
                            RMSE_gamma = np.mean((orien_est_all[:,2]-orient_GT_all[:,2]) ** 2)**(1/2)
                            differ = np.abs(orien_est_all[:,1]-orient_GT_all[:,1])
                            indx = differ<45
                            RMSE_phi2 = np.mean((differ[indx]) ** 2)**(1/2)

                            
                            #evaluate within the Jaccard circle
                            std_loc_thred = np.mean(((con_bias_x[index_all]-con_bias_x_all2)**2+(con_bias_y[index_all]-con_bias_y_all2)**2))**(1/2)
                            std_I_thred = np.mean((I_bias[index_all]**2))**(1/2)
                            con_bias_x_all_thred =np.mean(con_bias_x[index_all])
                            con_bias_y_all_thred = np.mean(con_bias_y[index_all])
                            I_bias_all_thred = np.mean(I_bias[index_all])
                        # log_output = {'Jaccard_1SM': jaccard,
                        #              'RMSE_loc_1SM':RMSE_loc,
                        #              'RMSE_I_1SM':RMSE_I}
                        if self.config['is_training_mode']==0:
                            self.writer.add_scalar({'Jaccard_1SM': jaccard})
                            self.writer.add_scalar({'std_loc_1SM': std_loc})
                            self.writer.add_scalar({'std_I_1SM': std_I})
                            self.writer.add_scalar({'loc_x_bias': con_bias_x_all})
                            self.writer.add_scalar({'loc_y_bias': con_bias_y_all})
                            self.writer.add_scalar({'I_bias': I_bias_all})

                            self.writer.add_scalar({'loc_x_bias_thred': con_bias_x_all2})
                            self.writer.add_scalar({'loc_y_bias_thred': con_bias_y_all2})
                            self.writer.add_scalar({'std_loc_1SM_thred': std_loc_thred})
                            self.writer.add_scalar({'std_I_1SM_thred': std_I_thred})
                            self.writer.add_scalar({'loc_x_bias_thred': con_bias_x_all_thred})
                            self.writer.add_scalar({'loc_y_bias_thred': con_bias_y_all_thred})
                            self.writer.add_scalar({'I_bias_thred': I_bias_all_thred})

                            self.writer.add_scalar({'RMSE_theta': RMSE_theta})
                            self.writer.add_scalar({'RMSE_phi': RMSE_phi})
                            self.writer.add_scalar({'RMSE_gamma': RMSE_gamma})
                            self.writer.add_scalar({'RMSE_phi2': RMSE_phi2})

                            self.writer.add_plot('1SM_loc', fig)
                            self.writer.add_plot('1SM_angles', fig_angles)
                            self.writer.add_plot('intesity_estimations', fig_I)
                            self.writer.add_plot('1SM_angles_thred', fig_angles_thred)
                            self.writer.add_plot('intesity_estimations_thred', fig_I_thred)
                            plt.close('all')

                        con_bias_x_all_epoches = np.concatenate((con_bias_x_all_epoches,con_bias_x_all),axis=0)
                        con_bias_y_all_epoches = np.concatenate((con_bias_y_all_epoches,con_bias_y_all),axis=0)
                        std_loc_epoches = np.concatenate((std_loc_epoches,std_loc),axis=0)
                        con_bias_x_all2_epoches = np.concatenate((con_bias_x_all2_epoches,con_bias_x_all2),axis=0)                       
                        con_bias_y_all2_epoches = np.concatenate((con_bias_y_all2_epoches,con_bias_y_all2),axis=0)
                        I_bias_all_epoches = np.concatenate((I_bias_all_epoches,I_bias_all),axis=0)
                        std_I_epoches = np.concatenate((std_I_epoches,std_I),axis=0)
                        RMSE_theta_epoches = np.concatenate((RMSE_theta_epoches,RMSE_theta),axis=0)
                        RMSE_phi_epoches = np.concatenate((RMSE_phi_epoches,RMSE_phi),axis=0)
                        RMSE_gamma_epoches = np.concatenate((RMSE_gamma_epoches,RMSE_gamma),axis=0)
                        RMSE_phi2_epoches = np.concatenate((RMSE_phi2_epoches,RMSE_phi2),axis=0)

                        
                        #evaluate within the Jaccard circle
                        std_loc_thred_epoches = np.concatenate((std_loc_thred_epoches,std_loc_thred),axis=0)
                        std_I_thred_epoches = np.concatenate((std_I_thred_epoches,std_I_thred),axis=0)
                        con_bias_x_all_thred_epoches = np.concatenate((con_bias_x_all_thred_epoches,con_bias_x_all_thred),axis=0)
                        con_bias_y_all_thred_epoches = np.concatenate((con_bias_y_all_thred_epoches,con_bias_y_all_thred),axis=0)
                        I_bias_all_thred_epoches = np.concatenate((I_bias_all_thred_epoches,I_bias_all_thred),axis=0)

                    
                
                        
                        
    
        
        sio.savemat("test_output.mat",{'Jaccard_1SM':jaccard,'RMSE_loc_1SM':std_loc,'RMSE_I_1SM':std_I,'loc_x_bias':con_bias_x_all,'loc_y_bias':con_bias_y_all,})
        return 