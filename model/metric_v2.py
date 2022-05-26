from cmath import nan
import torch
import torch.nn.functional as F
from model.loss import matlab_style_gauss2D
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def my_metric(output, target):
    with torch.no_grad():
        return F.mse_loss(output, target)


def my_metric2(output, target):
     with torch.no_grad():
        return F.l1_loss(output, target)

def jaccard_index(output, target, ce = False):
    eps = 1e-9
    
    bs = output.shape[0]
    if ce:
        pred = torch.argmax(output, 1).contiguous().view(bs, -1)
    else:
        pred = (output[:,0,:,:] > 1).contiguous().view(bs, -1)
    gt = target[:,1,:,:].contiguous().view(bs, -1)
    tp = torch.sum(pred * gt, 1)
    #print(tp, torch.sum(pred, 1), torch.sum(gt, 1))
    jaccard = torch.mean(tp / (torch.sum(pred, 1) + torch.sum(gt, 1) - tp + eps))

    return jaccard

def jaccard_index_2(output, target, ce = False):
    eps = 1e-9
    
    bs = output.shape[0]
    if ce:
        pred = torch.argmax(output, 1).contiguous().view(bs, -1)
        gt = target[:,1,:,:].contiguous().view(bs, -1)
    else:
        pred = output.contiguous().view(bs, -1)
        pred = pred / torch.max(pred)

        gt = target[:,0,:,:].unsqueeze(1)
        to_blur = matlab_style_gauss2D(shape=(7,7),sigma=1.0)
        to_blur = torch.from_numpy(to_blur).view(1, 1, 7, 7).to(pred.device)
        gt_blurred = F.conv2d(gt, to_blur, stride=1, padding=3).contiguous().view(bs, -1)
        gt_blurred = gt_blurred / torch.max(gt_blurred)

    # tp = torch.sum(gt_blurred * gt_blurred, 1)
    # print(torch.mean(tp / (torch.sum(gt_blurred, 1) + torch.sum(gt_blurred, 1) - tp + eps)))

    tp = torch.sum(pred * gt_blurred, 1)
    #print(tp, torch.sum(pred, 1), torch.sum(gt, 1))
    jaccard = torch.mean(tp / (torch.sum(pred, 1) + torch.sum(gt_blurred, 1) - tp + eps))

    return jaccard


def RMSE_1SM(config, output, target):
    
    MSE_loc = 0
    MSE_I = 0

    TP,FN,FP = 0,0,0
    count = 0
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']  # in unit of nm #put these parameters into config in the future
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    pixel_size = pixel_size_org/upsampling
    Jaccard_thred_pixel = config['Jaccard_1SM_thred']
    Jaccard_thred = Jaccard_thred_pixel*pixel_size  #


    has_SM = 0

    output = output.cpu().numpy()
    target = target.cpu().numpy()
    B,L,H,W = np.shape(output)

    x_GT = int(H/2)  # for 1SM case, I know all the GT's position is at (1,1)
    y_GT = int(W/2)   #in unit of pixel
    x_GT = np.expand_dims(x_GT,0)
    y_GT = np.expand_dims(y_GT,0)

    #pre_est = np.sum(output,axis=1)
    for ii in range(B):
        pre_est_cur = output[ii,:,:,:]
        I_GT = np.max(target[ii,0,:,:])
        I_thresh = 100
        x_est_save,y_est_save, est_img_crop = postprocessing_loc_v2(pre_est_cur, I_thresh)
        N_SM = np.size(x_est_save)
        if N_SM==0:
            FN+=1
        else:
            if has_SM==0:
                has_SM=1
                x_est_save_all=x_est_save
                y_est_save_all=y_est_save
                est_img_crop_all=est_img_crop
                N_SM_count = np.reshape(N_SM,(1,1))
                I_GT_all = np.ones((N_SM,1))*I_GT

                theta_GT_all = np.reshape(np.ones((N_SM,1))*target[ii,2,x_GT,y_GT],(N_SM,1))
                phi_GT_all = np.reshape(np.ones((N_SM,1))*target[ii,3,x_GT,y_GT],(N_SM,1))
                gamma_GT_all = np.reshape(np.ones((N_SM,1))*target[ii,4,x_GT,y_GT],(N_SM,1))
                

            else:               
                x_est_save_all=np.concatenate((x_est_save_all,x_est_save),axis=0)
                y_est_save_all=np.concatenate((y_est_save_all,y_est_save),axis=0)
                est_img_crop_all=np.concatenate((est_img_crop_all,est_img_crop),axis=0)
                N_SM_count = np.concatenate((N_SM_count,np.reshape(N_SM,(1,1))),axis=0)
                I_GT_all = np.concatenate((I_GT_all,np.ones((N_SM,1))*I_GT),axis=0)  

                theta_GT_all = np.concatenate((theta_GT_all,np.reshape(np.ones((N_SM,1))*target[ii,2,x_GT,y_GT],(N_SM,1))),axis=0)  
                phi_GT_all = np.concatenate((phi_GT_all,np.reshape(np.ones((N_SM,1))*target[ii,3,x_GT,y_GT],(N_SM,1))),axis=0)  
                gamma_GT_all = np.concatenate((gamma_GT_all,np.reshape(np.ones((N_SM,1))*target[ii,4,x_GT,y_GT],(N_SM,1))),axis=0)  


    

    if has_SM==0:
        RMSE_loc = 1e10
        RMSE_I = 1e10

        jaccard = 0
        loss = [jaccard,RMSE_loc,RMSE_I]
        loss_detail = [0,FN,0,1e10,1e10,0]
        bias_con_related = []
        orien_est = []
        orienta_GT = []
        M_est = []
    else:
        #calculate metric
        bias_con_x_all, bias_con_y_all,I_est_all, orien_est,M_est = loc_angle_est(config, est_img_crop_all, x_GT,y_GT, 
                                                                x_est_save_all,y_est_save_all)       
        MSE_loc_cur = bias_con_x_all**2+bias_con_y_all**2
        count+=np.sum(MSE_loc_cur<Jaccard_thred**2)
        TP +=np.sum(MSE_loc_cur<Jaccard_thred**2)
        MSE_loc+=np.sum(MSE_loc_cur[MSE_loc_cur<Jaccard_thred**2])
        MSE_I += np.sum((I_est_all[MSE_loc_cur<Jaccard_thred**2]-I_GT_all)**2)

        FP+=np.sum(MSE_loc_cur>=Jaccard_thred**2)

        RMSE_loc = (MSE_loc/count)**(1/2)
        RMSE_I = (MSE_I/count)**(1/2)
        # continuous error
        mean_con_bias_x = np.mean(bias_con_x_all)
        mean_con_bias_y = np.mean(bias_con_y_all)
        #std_con_bias_x = np.std(bias_con_x_all)
        #std_con_bias_y = np.std(bias_con_y_all)

        jaccard = TP/(TP+FN+FP)
        loss = [jaccard,RMSE_loc,RMSE_I]
        loss_detail = [TP,FN,FP,MSE_loc,MSE_I,count]
        #bias_con_related = [mean_con_bias_x, mean_con_bias_y, std_con_bias_x, std_con_bias_y]
        bias_con_related = np.concatenate((bias_con_x_all,bias_con_y_all),axis=1)
        orienta_GT = np.concatenate((theta_GT_all,phi_GT_all,gamma_GT_all),axis=1)

    return loss,loss_detail,bias_con_related,orien_est,orienta_GT,M_est,I_GT_all,I_est_all


def RMSE_1SM_resnet(config, output, target):
    
    

    output = output.cpu().numpy()
    target = target.cpu().numpy()

    orien_est,M_est,I_est_all = second2first_resnet(output)
    #bias_con_related = [mean_con_bias_x, mean_con_bias_y, std_con_bias_x, std_con_bias_y]
    N = np.int(np.size(output)/6)
    bias_con_x_all = np.zeros((N,1))
    bias_con_y_all = np.zeros((N,1))


    M_est = np.transpose(M_est)

    est = np.concatenate((I_est_all,I_est_all, bias_con_x_all,bias_con_y_all,orien_est,M_est),axis=1)
    return est



def second2first_resnet(output):

    XX_est = output[:,0]
    YY_est = output[:,1]
    ZZ_est = output[:,2]
    XY_est = output[:,3]
    XZ_est = output[:,4]
    YZ_est = output[:,5]

    N = np.size(XX_est)

    I_est = XX_est+YY_est+ZZ_est
    coeffcient = I_est
    coefficient = XX_est+YY_est+ZZ_est
    XX_est = XX_est/coefficient
    YY_est = YY_est/coefficient
    ZZ_est = ZZ_est/coefficient
    XY_est = XY_est/coefficient
    XZ_est = XZ_est/coefficient
    YZ_est = YZ_est/coefficient
    I_est = I_est/1.7726
    
    XX_est = np.reshape(XX_est,(1,1,N))
    YY_est = np.reshape(YY_est,(1,1,N))
    ZZ_est = np.reshape(ZZ_est,(1,1,N))
    XY_est = np.reshape(XY_est,(1,1,N))
    XZ_est = np.reshape(XZ_est,(1,1,N))
    YZ_est = np.reshape(YZ_est,(1,1,N))

    gamma = np.zeros((N,1))
    thetaD = np.zeros((N,1))
    phiD = np.zeros((N,1))
    M1 = np.concatenate((XX_est,XY_est,XZ_est),1)
    M2 = np.concatenate((XY_est,YY_est,YZ_est),1)
    M3 = np.concatenate((XZ_est,YZ_est,ZZ_est),1)
    M = np.concatenate((M1,M2,M3),0)
    for ii in range(N):
        [U,S,Vh] = np.linalg.svd(M[:,:,ii])
        mux = np.real(U[0,0])
        muy = np.real(U[1,0])
        muz = np.real(U[2,0])
        if muz<0:
           mux = -mux
           muy = -muy
           muz = -muz
        gamma[ii] = 1.5*np.real(S[0])-0.5
    
        thetaD[ii] = np.arccos(muz)/np.math.pi*180
        phiD[ii] = np.arctan2(muy,mux)/np.math.pi*180
    orien = np.concatenate((thetaD,phiD,gamma),axis=1)
    M_est = np.reshape(np.concatenate((XX_est,YY_est,ZZ_est,XY_est,XZ_est,YZ_est),1),(6,-1))

    return orien,M_est,np.reshape(I_est,(N,1))

def postprocessing(config, output,idx):
    

    count = 0
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']  # in unit of nm #put these parameters into config in the future
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    pixel_size = pixel_size_org/upsampling
    Jaccard_thred_pixel = config['Jaccard_1SM_thred']
    I_thresh = config['microscopy_params']['setup_params']['I_thresh']
    Jaccard_thred = Jaccard_thred_pixel*pixel_size  #


    has_SM = 0

    output = output.cpu().numpy()
    B,L,H,W = np.shape(output)

    x_GT = int(H/2)  # for 1SM case, I know all the GT's position is at (1,1)
    y_GT = int(W/2)   #in unit of pixel
    x_GT = np.expand_dims(x_GT,0)
    y_GT = np.expand_dims(y_GT,0)

    #pre_est = np.sum(output,axis=1)
    for ii in range(B):
        pre_est_cur = output[ii,:,:,:]
        #I_thresh = 50
        x_est_save,y_est_save, est_img_crop = postprocessing_loc_v2(pre_est_cur, I_thresh)
        N_SM = np.size(x_est_save)
        if N_SM==0:
           aaa=1
        else:
            if has_SM==0:
                has_SM=1
                x_est_save_all=x_est_save
                y_est_save_all=y_est_save
                est_img_crop_all=est_img_crop
                N_SM_count = np.reshape(N_SM,(1,1))
                idx_all = np.ones((N_SM,1))*idx[ii].cpu().numpy()

                

            else:               
                x_est_save_all=np.concatenate((x_est_save_all,x_est_save),axis=0)
                y_est_save_all=np.concatenate((y_est_save_all,y_est_save),axis=0)
                est_img_crop_all=np.concatenate((est_img_crop_all,est_img_crop),axis=0)
                N_SM_count = np.concatenate((N_SM_count,np.reshape(N_SM,(1,1))),axis=0)
                idx_all = np.concatenate((idx_all,np.ones((N_SM,1))*idx[ii].cpu().numpy()),axis=0)


    if has_SM==0:
        est = []
    else:
        #calculate metric
        bias_con_x_all, bias_con_y_all,I_est_all, orien_est,M_est = loc_angle_est(config, est_img_crop_all, x_GT,y_GT, 
                                                                x_est_save_all,y_est_save_all)       
        

        M_est = np.transpose(M_est)
        #bias_con_related = [mean_con_bias_x, mean_con_bias_y, std_con_bias_x, std_con_bias_y]
        est = np.concatenate((idx_all,I_est_all, bias_con_x_all,bias_con_y_all,orien_est,M_est),axis=1)
        
    return est


def postprocessingv2(config, outputGT, output, idx):
    

    count = 0
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']  # in unit of nm #put these parameters into config in the future
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    pixel_size = pixel_size_org/upsampling
    Jaccard_thred_pixel = config['Jaccard_1SM_thred']
    I_thresh = config['microscopy_params']['setup_params']['I_thresh']
    Jaccard_thred = Jaccard_thred_pixel*pixel_size  #


    has_SM = 0

    output = output.cpu().detach().numpy()
    B,L,H,W = np.shape(output)

    x_GT = int(H/2)  # for 1SM case, I know all the GT's position is at (1,1)
    y_GT = int(W/2)   #in unit of pixel
    x_GT = np.expand_dims(x_GT,0)
    y_GT = np.expand_dims(y_GT,0)

    #pre_est = np.sum(output,axis=1)
    for ii in range(B):
        pre_est_cur = output[ii,:,:,:]
        #I_thresh = 50
        x_est_save,y_est_save, est_img_crop = postprocessing_loc_v2(pre_est_cur, I_thresh)
        N_SM = np.size(x_est_save)
        if N_SM==0:
           aaa=1
        else:
            if has_SM==0:
                has_SM=1
                x_est_save_all=x_est_save
                y_est_save_all=y_est_save
                est_img_crop_all=est_img_crop
                N_SM_count = np.reshape(N_SM,(1,1))
                idx_all = np.ones((N_SM,1))*idx[ii].cpu().numpy()

                

            else:               
                x_est_save_all=np.concatenate((x_est_save_all,x_est_save),axis=0)
                y_est_save_all=np.concatenate((y_est_save_all,y_est_save),axis=0)
                est_img_crop_all=np.concatenate((est_img_crop_all,est_img_crop),axis=0)
                N_SM_count = np.concatenate((N_SM_count,np.reshape(N_SM,(1,1))),axis=0)
                idx_all = np.concatenate((idx_all,np.ones((N_SM,1))*idx[ii].cpu().numpy()),axis=0)


    if has_SM==0:
        est = []
    else:
        #calculate metric
        bias_con_x_all, bias_con_y_all,I_est_all, orien_est,M_est = loc_angle_est(config, est_img_crop_all, x_GT,y_GT, 
                                                                x_est_save_all,y_est_save_all)       
        

        M_est = np.transpose(M_est)
        #bias_con_related = [mean_con_bias_x, mean_con_bias_y, std_con_bias_x, std_con_bias_y]
        est = np.concatenate((idx_all,I_est_all, bias_con_x_all,bias_con_y_all,orien_est,M_est),axis=1)
        
    return est



# Find continuous gt & prediction location
def find_continuous_bias(config, target_map, gt_loc, prediction_map, pred_loc, n_sm = 1):
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    pixel_size = pixel_size_org/upsampling
    
    x_pred, y_pred = pred_loc
    x_gt, y_gt = gt_loc
    # First we index maps
    H,W = target_map.shape
    index_H_map = np.arange(H)
    index_W_map = np.arange(W)
    # Now we gaussian find location
    if n_sm == 1:
        if np.size(x_pred) != 1:
            warnings.warn("False positive in validaion 1SM")
        
        pred_crop = prediction_map[int(x_pred[0])-8:int(x_pred[0])+9, int(y_pred[0])-8:int(y_pred[0])+9]
        target_crop = target_map[int(x_gt)-8:int(x_gt)+9, int(y_gt)-8:int(y_gt)+9]
        # Make the summation 1
        pred_crop = pred_crop / np.sum(pred_crop)
        target_crop = target_crop / np.sum(target_crop)
        
        pred_H_crop = pred_crop.sum(1)
        pred_W_crop = pred_crop.sum(0)
        index_pred_H_crop = index_H_map[int(x_pred[0])-8:int(x_pred[0])+9]
        index_pred_W_crop = index_W_map[int(y_pred[0])-8:int(y_pred[0])+9]
        
        target_H_crop = target_crop.sum(1)
        target_W_crop = target_crop.sum(0)
        index_target_H_crop = index_H_map[int(x_gt)-8:int(x_gt)+9]
        index_target_W_crop = index_W_map[int(y_gt)-8:int(y_gt)+9]

        # Find x, y location for both
        x_post_pred = np.sum(pred_H_crop * index_pred_H_crop) 
        y_post_pred = np.sum(pred_W_crop * index_pred_W_crop) 
        x_post_gt = np.sum(target_H_crop * index_target_H_crop) 
        y_post_gt = np.sum(target_W_crop * index_target_W_crop) 
        #print("xy",[x_post_pred,y_post_pred],[x_post_gt,y_post_gt])
        bias_x = (x_post_pred-x_post_gt)*pixel_size
        bias_y = (y_post_pred-y_post_gt)*pixel_size
        
        return bias_x, bias_y
    elif n_sm == 2:
        bias_x_center = np.nan
        bias_y_center = np.nan
        bias_x_around = np.nan
        bias_y_around = np.nan
        if np.size(x_pred) > 2:
            warnings.warn("False positive in validaion 2SMs")
        for i in range(np.size(x_pred)):
            distance_1 = np.abs(x_pred[i] - x_gt[0]) + np.abs(y_pred[i] - y_gt[0])
            distance_2 = np.abs(x_pred[i] - x_gt[1]) + np.abs(y_pred[i] - y_gt[1])
            if distance_1 < distance_2:
                # Center case
                center = True
                x_gt_select = int(x_gt[0])
                y_gt_select = int(y_gt[0])
            else:
                center = False
                x_gt_select = int(x_gt[1])
                y_gt_select = int(y_gt[1])
            
            pred_crop = prediction_map[int(x_pred[i])-3:int(x_pred[i])+4, int(y_pred[i])-3:int(y_pred[i])+4]
            target_crop = target_map[x_gt_select-3:x_gt_select+4, y_gt_select-3:y_gt_select+4]
            # Make the summation 1
            pred_crop = pred_crop / np.sum(pred_crop)
            target_crop = target_crop / np.sum(target_crop)
            
            pred_H_crop = pred_crop.sum(1)
            pred_W_crop = pred_crop.sum(0)
            index_pred_H_crop = index_H_map[int(x_pred[i])-3:int(x_pred[i])+4]
            index_pred_W_crop = index_W_map[int(y_pred[i])-3:int(y_pred[i])+4]

            target_H_crop = target_crop.sum(1)
            target_W_crop = target_crop.sum(0)
            index_target_H_crop = index_H_map[x_gt_select-3:x_gt_select+4]
            index_target_W_crop = index_W_map[y_gt_select-3:y_gt_select+4]
            
            # Find x, y location for both
            x_post_pred = np.sum(pred_H_crop * index_pred_H_crop) 
            y_post_pred = np.sum(pred_W_crop * index_pred_W_crop) 
            x_post_gt = np.sum(target_H_crop * index_target_H_crop) 
            y_post_gt = np.sum(target_W_crop * index_target_W_crop) 

            if center:
                bias_x_center = (x_post_pred-x_post_gt)*pixel_size
                bias_y_center = (y_post_pred-y_post_gt)*pixel_size
            else:
                bias_x_around = (x_post_pred-x_post_gt)*pixel_size
                bias_y_around = (y_post_pred-y_post_gt)*pixel_size
        return bias_x_center, bias_y_center, bias_x_around, bias_y_around
            
def loc_angle_est(config, crop_est_images, x_gt,y_gt,x_pred,y_pred):
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    pixel_size = pixel_size_org/upsampling
    rad = 3

    HW_grid = np.meshgrid(np.arange(rad*2+1)-rad,np.arange(rad*2+1)-rad)
    

    XX_est_all = crop_est_images[:,0,:,:]
    YY_est_all = crop_est_images[:,1,:,:]
    ZZ_est_all = crop_est_images[:,2,:,:]
    XY_est_all = crop_est_images[:,3,:,:]
    XZ_est_all = crop_est_images[:,4,:,:]
    YZ_est_all = crop_est_images[:,5,:,:]
    I_est_all = XX_est_all+YY_est_all+ZZ_est_all
    N = np.shape(YZ_est_all)[0]
    I_sum = np.reshape(np.sum(I_est_all,(1,2)),(N,1,1))

    I_temp = I_est_all.copy()
    #I_temp[I_temp<np.reshape(0.1*np.amax(np.amax(I_temp,axis=-1),axis=-1),(N,1,1))]=np.nan
    bais_imagesx = I_temp*HW_grid[0]/np.reshape(np.sum(I_est_all,axis=(1,2)),(N,1,1))
    bais_imagesy = I_temp*HW_grid[1]/np.reshape(np.sum(I_est_all,axis=(1,2)),(N,1,1))
    
    bias_x = (np.reshape(np.sum(bais_imagesx,(1,2)),(N,1))+x_pred-x_gt)*pixel_size
    bias_y = (np.reshape(np.sum(bais_imagesy,(1,2)),(N,1))+y_pred-y_gt)*pixel_size

    g_model = I_est_all/np.reshape(np.max(I_est_all,axis=(1,2)),(-1,1,1))
    XX_est = np.sum(XX_est_all,axis=(1,2))
    YY_est = np.sum(YY_est_all,axis=(1,2))
    ZZ_est = np.sum(ZZ_est_all,axis=(1,2))
    XY_est = np.sum(XY_est_all,axis=(1,2))
    XZ_est = np.sum(XZ_est_all,axis=(1,2))
    YZ_est = np.sum(YZ_est_all,axis=(1,2))
    # XX_est = np.sum(XX_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    # YY_est = np.sum(YY_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    # ZZ_est = np.sum(ZZ_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    # XY_est = np.sum(XY_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    # XZ_est = np.sum(XZ_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    # YZ_est = np.sum(YZ_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    I_est = np.sqrt(np.sum(I_est_all*I_est_all,axis=(1,2)))
    coefficient = XX_est+YY_est+ZZ_est
    XX_est = XX_est/coefficient
    YY_est = YY_est/coefficient
    ZZ_est = ZZ_est/coefficient
    XY_est = XY_est/coefficient
    XZ_est = XZ_est/coefficient
    YZ_est = YZ_est/coefficient
    I_est = I_est/1.7726

#  second method to calculate the second moment
    # XX_est = XX_est_all
    # YY_est = YY_est_all
    # ZZ_est = ZZ_est_all
    # XY_est = XY_est_all
    # XZ_est = XZ_est_all
    # YZ_est = YZ_est_all
    # coefficient = XX_est+YY_est+ZZ_est
    # XX_est = XX_est/coefficient
    # XX_est = np.minimum(np.maximum(XX_est,0),1)
    # YY_est = YY_est/coefficient
    # YY_est = np.minimum(np.maximum(YY_est,0),1)
    # ZZ_est = ZZ_est/coefficient
    # ZZ_est = np.minimum(np.maximum(ZZ_est,0),1)
    # XY_est = XY_est/coefficient
    # XY_est = np.minimum(np.maximum(XY_est,-0.5),0.5)
    # XZ_est = XZ_est/coefficient
    # XZ_est = np.minimum(np.maximum(XZ_est,-0.5),0.5)
    # YZ_est = YZ_est/coefficient
    # YZ_est = np.minimum(np.maximum(YZ_est,-0.5),0.5)

    # g_model = I_est_all/np.reshape(np.max(I_est_all,axis=(1,2)),(-1,1,1))
    # g_model[g_model<0.2]=0

    # XX_est = np.sum(XX_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # YY_est = np.sum(YY_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # ZZ_est = np.sum(ZZ_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # XY_est = np.sum(XY_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # XZ_est = np.sum(XZ_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # YZ_est = np.sum(YZ_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))

    # coefficient = XX_est+YY_est+ZZ_est
    # XX_est = XX_est/coefficient
    # YY_est = YY_est/coefficient
    # ZZ_est = ZZ_est/coefficient
    # XY_est = XY_est/coefficient
    # XZ_est = XZ_est/coefficient
    # YZ_est = YZ_est/coefficient


    # I_est = np.sqrt(np.sum(I_est_all*I_est_all,axis=(1,2)))
    # I_est = I_est/1.7726
    #-----------------------------------------------------------
    
    XX_est = np.reshape(XX_est,(1,1,N))
    YY_est = np.reshape(YY_est,(1,1,N))
    ZZ_est = np.reshape(ZZ_est,(1,1,N))
    XY_est = np.reshape(XY_est,(1,1,N))
    XZ_est = np.reshape(XZ_est,(1,1,N))
    YZ_est = np.reshape(YZ_est,(1,1,N))

    gamma = np.zeros((N,1))
    thetaD = np.zeros((N,1))
    phiD = np.zeros((N,1))
    M1 = np.concatenate((XX_est,XY_est,XZ_est),1)
    M2 = np.concatenate((XY_est,YY_est,YZ_est),1)
    M3 = np.concatenate((XZ_est,YZ_est,ZZ_est),1)
    M = np.concatenate((M1,M2,M3),0)
    for ii in range(N):
        [U,S,Vh] = np.linalg.svd(M[:,:,ii])
        mux = np.real(U[0,0])
        muy = np.real(U[1,0])
        muz = np.real(U[2,0])
        if muz<0:
           mux = -mux
           muy = -muy
           muz = -muz
        gamma[ii] = 1.5*np.real(S[0])-0.5
    
        thetaD[ii] = np.arccos(muz)/np.math.pi*180
        phiD[ii] = np.arctan2(muy,mux)/np.math.pi*180
    orien = np.concatenate((thetaD,phiD,gamma),axis=1)
    M_est = np.reshape(np.concatenate((XX_est,YY_est,ZZ_est,XY_est,XZ_est,YZ_est),1),(6,-1))
           
    return bias_x, bias_y,np.reshape(I_est,(N,1)), orien, M_est


def loc_angle_est2(config, crop_est_images, x_gt,y_gt,x_pred,y_pred):
    pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org']
    upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
    pixel_size = pixel_size_org/upsampling
    rad = 3

    HW_grid = np.meshgrid(np.arange(rad*2+1)-rad,np.arange(rad*2+1)-rad)
    

    XX_est_all = crop_est_images[:,0,:,:]
    YY_est_all = crop_est_images[:,1,:,:]
    ZZ_est_all = crop_est_images[:,2,:,:]
    XY_est_all = crop_est_images[:,3,:,:]
    XZ_est_all = crop_est_images[:,4,:,:]
    YZ_est_all = crop_est_images[:,5,:,:]
    gamma_est_all = crop_est_images[:,6,:,:]
    I_est_all = XX_est_all+YY_est_all+ZZ_est_all
    N = np.shape(YZ_est_all)[0]
    I_sum = np.reshape(np.sum(I_est_all,(1,2)),(N,1,1))

    I_temp = I_est_all.copy()
    #I_temp[I_temp<np.reshape(0.1*np.amax(np.amax(I_temp,axis=-1),axis=-1),(N,1,1))]=np.nan
    bais_imagesx = I_temp*HW_grid[0]/np.reshape(np.sum(I_est_all,axis=(1,2)),(N,1,1))
    bais_imagesy = I_temp*HW_grid[1]/np.reshape(np.sum(I_est_all,axis=(1,2)),(N,1,1))
    
    bias_x = (np.reshape(np.sum(bais_imagesx,(1,2)),(N,1))+x_pred-x_gt)*pixel_size
    bias_y = (np.reshape(np.sum(bais_imagesy,(1,2)),(N,1))+y_pred-y_gt)*pixel_size

    g_model = I_est_all/np.reshape(np.max(I_est_all,axis=(1,2)),(-1,1,1))
    XX_est = np.sum(XX_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    YY_est = np.sum(YY_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    ZZ_est = np.sum(ZZ_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    XY_est = np.sum(XY_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    XZ_est = np.sum(XZ_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    YZ_est = np.sum(YZ_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))
    gamma_est = 1-np.sum(gamma_est_all*g_model,axis=(1,2))/np.sum(g_model*g_model,axis=(1,2))/1000*3
    I_est = np.sqrt(np.sum(I_est_all*I_est_all,axis=(1,2)))
    coefficient = XX_est+YY_est+ZZ_est
    XX_est = XX_est/coefficient
    YY_est = YY_est/coefficient
    ZZ_est = ZZ_est/coefficient
    XY_est = XY_est/coefficient
    XZ_est = XZ_est/coefficient
    YZ_est = YZ_est/coefficient
    I_est = I_est/1.7726

#  second method to calculate the second moment
    # XX_est = XX_est_all
    # YY_est = YY_est_all
    # ZZ_est = ZZ_est_all
    # XY_est = XY_est_all
    # XZ_est = XZ_est_all
    # YZ_est = YZ_est_all
    # coefficient = XX_est+YY_est+ZZ_est
    # XX_est = XX_est/coefficient
    # XX_est = np.minimum(np.maximum(XX_est,0),1)
    # YY_est = YY_est/coefficient
    # YY_est = np.minimum(np.maximum(YY_est,0),1)
    # ZZ_est = ZZ_est/coefficient
    # ZZ_est = np.minimum(np.maximum(ZZ_est,0),1)
    # XY_est = XY_est/coefficient
    # XY_est = np.minimum(np.maximum(XY_est,-0.5),0.5)
    # XZ_est = XZ_est/coefficient
    # XZ_est = np.minimum(np.maximum(XZ_est,-0.5),0.5)
    # YZ_est = YZ_est/coefficient
    # YZ_est = np.minimum(np.maximum(YZ_est,-0.5),0.5)

    # g_model = I_est_all/np.reshape(np.max(I_est_all,axis=(1,2)),(-1,1,1))
    # g_model[g_model<0.2]=0

    # XX_est = np.sum(XX_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # YY_est = np.sum(YY_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # ZZ_est = np.sum(ZZ_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # XY_est = np.sum(XY_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # XZ_est = np.sum(XZ_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))
    # YZ_est = np.sum(YZ_est*g_model,axis=(1,2))/np.sum(g_model,axis=(1,2))

    # coefficient = XX_est+YY_est+ZZ_est
    # XX_est = XX_est/coefficient
    # YY_est = YY_est/coefficient
    # ZZ_est = ZZ_est/coefficient
    # XY_est = XY_est/coefficient
    # XZ_est = XZ_est/coefficient
    # YZ_est = YZ_est/coefficient


    # I_est = np.sqrt(np.sum(I_est_all*I_est_all,axis=(1,2)))
    # I_est = I_est/1.7726
    #-----------------------------------------------------------
    
    XX_est = np.reshape(XX_est,(1,1,N))
    YY_est = np.reshape(YY_est,(1,1,N))
    ZZ_est = np.reshape(ZZ_est,(1,1,N))
    XY_est = np.reshape(XY_est,(1,1,N))
    XZ_est = np.reshape(XZ_est,(1,1,N))
    YZ_est = np.reshape(YZ_est,(1,1,N))

    gamma = np.zeros((N,1))
    thetaD = np.zeros((N,1))
    phiD = np.zeros((N,1))
    M1 = np.concatenate((XX_est,XY_est,XZ_est),1)
    M2 = np.concatenate((XY_est,YY_est,YZ_est),1)
    M3 = np.concatenate((XZ_est,YZ_est,ZZ_est),1)
    M = np.concatenate((M1,M2,M3),0)
    for ii in range(N):
        [U,S,Vh] = np.linalg.svd(M[:,:,ii])
        mux = np.real(U[0,0])
        muy = np.real(U[1,0])
        muz = np.real(U[2,0])
        if muz<0:
           mux = -mux
           muy = -muy
           muz = -muz
        #gamma[ii] = 1.5*np.real(S[0])-0.5
        gamma[ii] = gamma_est[ii]
    
        thetaD[ii] = np.arccos(muz)/np.math.pi*180
        phiD[ii] = np.arctan2(muy,mux)/np.math.pi*180
    orien = np.concatenate((thetaD,phiD,gamma),axis=1)
    M_est = np.reshape(np.concatenate((XX_est,YY_est,ZZ_est,XY_est,XZ_est,YZ_est),1),(6,-1))
           
    return bias_x, bias_y,np.reshape(I_est,(N,1)), orien, M_est

# For 2sm case
def RMSE_2SM(config, loc_out, ang_out, target, ce = False, ang_target = None):
    if config["train_loss"] != "localization_loss":
        if ang_target == None:
            ang_target = target
        theta_comp = val_theta(ang_out,ang_target)
    else:
        theta_comp = []
    
    if config["train_loss"] != "theta_loss":
        MSE_loc = 0
        MSE_I = 0

        TP,FN,FP = 0,0,0
        count = 0
        pixel_size_org = config['microscopy_params']['setup_params']['pixel_sz_org'] 
        upsampling = config['microscopy_params']['setup_params']['upsampling_ratio'] 
        pixel_size = pixel_size_org/upsampling
        Jaccard_thred_pixel = config['Jaccard_2SMs_thred']
        Jaccard_thred = Jaccard_thred_pixel*pixel_size  #

        bias_x_list_center = []
        bias_y_list_center = []
        bias_x_list_around = []
        bias_y_list_around = []

        bias_con_x_list_center = []
        bias_con_y_list_center = []
        bias_con_x_list_around = []
        bias_con_y_list_around = []

        loc_out = loc_out.cpu().numpy()
        target = target.cpu().numpy()
        B,L,H,W = np.shape(loc_out)
        filter_loc = matlab_style_gauss2D(shape=(7,7),sigma=1.0)
        radius_loc = [3,3] #half of [7,7]
        for ii in range(B):
            x_GT = np.zeros([2])
            y_GT = np.zeros([2])
            x_GT[0] = int(H/2)
            y_GT[0] = int(W/2)

            pre_est = loc_out[ii,0,:,:]
            I_GT = target[ii,0,:,:]
            x_GT_found, y_GT_found = find_gt(I_GT) # find ground truth location index
            # Check for order of the found gt
            if int(x_GT_found.flatten()[0]) == x_GT[0]:
                x_GT[1] = int(x_GT_found.flatten()[1])
                y_GT[1] = int(y_GT_found.flatten()[1])
            elif int(x_GT_found.flatten()[1]) == x_GT[0]:
                x_GT[1] = int(x_GT_found.flatten()[0])
                y_GT[1] = int(y_GT_found.flatten()[0])
            else:
                raise ValueError("Correct Location not found")

            I_thresh = 0.3*np.min([I_GT[int(x_GT[0]), int(y_GT[0])], I_GT[int(x_GT[1]), int(y_GT[1])]])
            I_est_save,x_est_save,y_est_save = postprocessing_loc(pre_est, I_thresh, radius_loc, filter_loc)

            N_SM = np.size(x_est_save)
            if N_SM == 0:
                FN += 2
            else:
                bias_con_x_center, bias_con_y_center, bias_con_x_around, bias_con_y_around = find_continuous_bias(config, target[ii,-1,:,:], 
                                                                                                  [x_GT,y_GT], pre_est, 
                                                                                                  [x_est_save,y_est_save], n_sm = 2)
                bias_con_x_list_center.append(bias_con_x_center)
                bias_con_y_list_center.append(bias_con_y_center)
                bias_con_x_list_around.append(bias_con_x_around)
                bias_con_y_list_around.append(bias_con_x_around)

                if N_SM == 1:
                    FN += 1
                for ii in range(N_SM):
                    TP_this_time = 0
                    possible_mse_list = []
                    for jj in range(x_GT.shape[0]):
                        possible_mse_list.append(((x_est_save[ii]-x_GT[jj])*pixel_size)**2+((y_est_save[ii]-y_GT[jj])*pixel_size)**2)
                    MSE_loc_cur = np.min(possible_mse_list)
                    which_gt = np.argmin(possible_mse_list)
                    if which_gt == 0:
                        bias_x_list_center.append((x_est_save[ii]-x_GT[which_gt])*pixel_size)
                        bias_y_list_center.append((y_est_save[ii]-y_GT[which_gt])*pixel_size)
                    else:
                        bias_x_list_around.append((x_est_save[ii]-x_GT[which_gt])*pixel_size)
                        bias_y_list_around.append((y_est_save[ii]-y_GT[which_gt])*pixel_size)

                    if MSE_loc_cur<Jaccard_thred**2:
                        count+=1

                        TP+=1
                        TP_this_time += 1
                        MSE_loc += MSE_loc_cur
                        MSE_I += (I_est_save[ii]-np.max(I_GT))**2
                    else:
                        FP+=1
                if TP_this_time > 2:
                    TP -= TP_this_time - 2
                    FP += TP_this_time - 2

        if count==0:
            RMSE_loc = 1e10
            RMSE_I = 1e10
        else:
            RMSE_loc = (MSE_loc/count)**(1/2)
            RMSE_I = (MSE_I/count)**(1/2)

        mean_bias_x_center = np.mean(bias_x_list_center)
        mean_bias_y_center = np.mean(bias_y_list_center)
        std_bias_x_center = np.std(bias_x_list_center)
        std_bias_y_center = np.std(bias_y_list_center)

        mean_bias_x_around = np.mean(bias_x_list_around)
        mean_bias_y_around = np.mean(bias_y_list_around)
        std_bias_x_around = np.std(bias_x_list_around)
        std_bias_y_around = np.std(bias_y_list_around)

        # Cont.
        mean_con_bias_x_center = np.nanmean(bias_con_x_list_center)
        mean_con_bias_y_center = np.nanmean(bias_con_y_list_center)
        std_con_bias_x_center = np.nanstd(bias_con_x_list_center)
        std_con_bias_y_center = np.nanstd(bias_con_y_list_center)

        mean_con_bias_x_around = np.nanmean(bias_con_x_list_around)
        mean_con_bias_y_around = np.nanmean(bias_con_y_list_around)
        std_con_bias_x_around = np.nanstd(bias_con_x_list_around)
        std_con_bias_y_around = np.nanstd(bias_con_y_list_around)

        jaccard = TP/(TP+FN+FP)
        loss = [jaccard,RMSE_loc,RMSE_I]
        loss_detail = [TP,FN,FP,MSE_loc,MSE_I,count]
        bias_related = [mean_bias_x_center, mean_bias_y_center, std_bias_x_center, std_bias_y_center, 
                        bias_x_list_center, bias_y_list_center,
                        mean_bias_x_around, mean_bias_y_around, std_bias_x_around, std_bias_y_around, 
                        bias_x_list_around, bias_y_list_around]
        bias_con_related = [mean_con_bias_x_center, mean_con_bias_y_center, std_con_bias_x_center, std_con_bias_y_center, 
                        bias_con_x_list_center, bias_con_y_list_center,
                        mean_con_bias_x_around, mean_con_bias_y_around, std_con_bias_x_around, std_con_bias_y_around, 
                        bias_con_x_list_around, bias_con_y_list_around]
    else:
        loss = []
        loss_detail = []
        bias_related = []
        bias_con_related = []
    return loss,loss_detail,bias_related,bias_con_related,theta_comp

def postprocessing_loc_v2(est_images, I_thresh):
    # pre_est: output image from the network
    channels = np.size(est_images,axis=0)
    rad = 3
    I_img = np.sum(est_images[0:3,:,:],axis=0)
    I_mask = I_img > I_thresh
    mask_label = label(I_mask)
    #print(np.sum(mask_label != 0))
    [H,W]=np.shape(I_img)
    N_SM = np.max(mask_label)
    if N_SM==0:
        x_est_save = []
        y_est_save = []
        est_img_crop = []
    else:
        x_est_save = np.zeros((N_SM,1)) 
        x_est_save[:]=np.NaN       
        y_est_save = np.zeros((N_SM,1))
        y_est_save[:]=np.NaN   
        est_img_crop = np.zeros((N_SM,channels,2*rad+1,2*rad+1))
        est_img_crop[:]=np.NaN   
        
        
        #print(np.max(mask_label))
        for ii in range(0, np.max(mask_label)):
            #k = np.argwhere(mask_label == ii)
            k = mask_label == ii+1
            # Find the position with the max intensity
            I_img_tmp = I_img.copy()
            I_img_tmp[~k] = 0
            indx_max = (I_img_tmp == np.max(I_img[k]))
            #print(indx_max)
            # in each block, use the pixel with the maximum intensity estimation as the estimated x,y locations
            x_est, y_est = np.argwhere(indx_max == 1)[0,:]#.squeeze()
            #print(x_est, y_est)
            if x_est>rad and x_est+rad<W and y_est>rad and y_est+rad<H:

                x_est_save[ii] = y_est
                y_est_save[ii] = x_est

                est_img_crop[ii,:,:,:] = est_images[:,x_est-rad:x_est+rad+1,y_est-rad:y_est+rad+1]
               
            
        est_img_crop = est_img_crop[~np.isnan(est_img_crop)]
        est_img_crop = np.reshape(est_img_crop,(-1,channels,2*rad+1,2*rad+1))
        x_est_save = x_est_save[~np.isnan(x_est_save)]
        x_est_save = np.reshape(x_est_save,(-1,1))
        y_est_save = y_est_save[~np.isnan(y_est_save)]
        y_est_save = np.reshape(y_est_save,(-1,1))

           
    
    return x_est_save,y_est_save, est_img_crop


def postprocessing_loc(pre_est, I_thresh, radius_loc, filter_loc):
    # pre_est: output image from the network
    image_size = np.shape(pre_est)
    I_img = pre_est
    I_img = signal.convolve2d(I_img, filter_loc, mode='same')
    I_mask = I_img > I_thresh
    mask_label = label(I_mask)
    #print(np.sum(mask_label != 0))
    
    N_SM = np.max(mask_label)
    if N_SM==0:
        I_est_save = []
        x_est_save = []
        y_est_save = []
    else:
        I_est_save = np.zeros((N_SM,1))
        x_est_save = np.zeros((N_SM,1))
        y_est_save = np.zeros((N_SM,1))
        
        # for intensity estimation, only use the center ~5*5 region from original 7*7 filter. 
        # name this new region as trust region
        size_filter_loc = filter_loc.shape
        trust_region_loc_min = 1
        trust_region_loc_max = size_filter_loc[0] - 1
        
        
        filter_loc_sum = np.sum(filter_loc[trust_region_loc_min:trust_region_loc_max, trust_region_loc_min:trust_region_loc_max])
        
        #print(np.max(mask_label))
        for ii in range(1, np.max(mask_label) + 1):
            #k = np.argwhere(mask_label == ii)
            k = mask_label == ii
            # Find the position with the max intensity
            I_img_tmp = I_img.copy()
            I_img_tmp[~k] = 0
            indx_max = (I_img_tmp == np.max(I_img[k]))
            #print(indx_max)
            # in each block, use the pixel with the maximum intensity estimation as the estimated x,y locations
            x_est, y_est = np.argwhere(indx_max == 1)[0,:]#.squeeze()
            #print(x_est, y_est)
            # use the estimated x,y location as the center, crop a block(7*7) that same as the gfilter size
            # the 'max', 'min' are for cases that  croped the 7*7 block falls out of the whole image
            x_begin = max(x_est - radius_loc[0], 0)
            x_end = min(x_est + radius_loc[0], image_size[0] - 1)
            y_begin = max(y_est - radius_loc[1], 0)
            y_end = min(y_est + radius_loc[1], image_size[1] - 1)
            I_est_matrix = I_img[x_begin: x_end + 1, y_begin:y_end + 1]
            #print(x_begin, x_end, y_begin, y_end, x_est, y_est, image_size)
            # convolve each block with gfilter as we did in Network
            
            # use the convolved value to estimate the intensity
            I_est = np.sum(I_est_matrix[trust_region_loc_min:trust_region_loc_max, trust_region_loc_min:trust_region_loc_max]) / filter_loc_sum
                    
            # put the post processed intensity value back to the estimation location pixel
            I_est_save[ii-1]=I_est
        
            x_est_save[ii-1] = x_est
            y_est_save[ii-1] = y_est
    
    return I_est_save,x_est_save,y_est_save

def val_theta(output, target):
    #pre_ests, I_thresh, radius_loc, radius_ang, filter_loc, filter_ang
    theta_pred_list = []
    theta_gt_list = []
    
    output = output.cpu()
    target = target.cpu()

    shape_ang = np.array([7,7])
    radius_ang = (shape_ang - 1) // 2
    filter_ang = torch.from_numpy(np.ones(shape_ang)).float()
    
    for i in range(output.shape[0]):
        pre_ests = output[i]
        target_img = target[i,0]
        image_size = pre_ests.shape[1:]
        theta_img = torch.squeeze(pre_ests[0,:,:]) # Changed to 0 for only theta
    
        # Directly use label on target
        mask_label = label(target_img)
    
        theta_est_img = torch.zeros(theta_img.shape)
        x_est_save = torch.zeros(15,1)
        y_est_save = torch.zeros(15,1)
    
        size_filter_ang = filter_ang.shape
        trust_region_ang = torch.arange(2, size_filter_ang[0] - 1)
    
        filter_ang_sum = torch.sum(filter_ang[trust_region_ang[0]:trust_region_ang[-1], trust_region_ang[0]:trust_region_ang[-1]])
    
        for ii in range(1, np.max(mask_label) + 1):
            k = torch.from_numpy(mask_label == ii)
            # Find the position with the max intensity
            target_img_tmp = target_img.clone()
            target_img_tmp[~k] = 0
            indx_max = (target_img_tmp == torch.max(target_img[k])).int()
            # in each block, use the pixel with the maximum intensity estimation as the estimated x,y locations
            x_gt, y_gt = np.argwhere(indx_max == 1)[:,0]
            # use the gt x,y location as the center, crop a block(7*7) that same as the gfilter size
            # the 'max', 'min' are for cases that  croped the 7*7 block falls out of the whole image
            x_begin = max(x_gt - radius_ang[0], 0)
            x_end = min(x_gt + radius_ang[0], image_size[0] - 1)
            y_begin = max(y_gt - radius_ang[1], 0)
            y_end = min(y_gt + radius_ang[1], image_size[1] - 1)
            theta_est_matrix = theta_img[x_begin:x_end+1, y_begin:y_end+1]
            theta_est = torch.sum(theta_est_matrix[trust_region_ang[0]:trust_region_ang[-1], trust_region_ang[0]:trust_region_ang[-1]]) / filter_ang_sum

            theta_pred_list.append(theta_est)
            theta_gt_list.append(target[i,2,x_gt,y_gt])
    
    return [theta_pred_list, theta_gt_list]
