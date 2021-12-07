import torch.nn.functional as F
import torch
from parse_config import ConfigParser
import torch.nn as nn
import numpy as np
import math

def cross_entropy(output, target):
    # 2nd channel in target is intensity mask
    weight = torch.tensor([1, 4000]).float().to(output.device)
    return F.cross_entropy(output, target[:,1,:,:].long(), weight = weight)

def mse_dice(output, target, scaling_factor):
    device = output.device
    bs = output.shape[0]

    psf_heatmap2 = matlab_style_gauss2D(shape=(7,7),sigma=1.0)
    gfilter2 = torch.from_numpy(psf_heatmap2).view(1, 1, 7, 7).to(device)

    predicted_blur = F.conv2d(output, gfilter2, stride=1, padding=3)

    target_mask = target[:,0,:,:].unsqueeze(1) 
    target_blur = F.conv2d(target_mask, gfilter2, stride=1, padding=3)

    mse_location = F.mse_loss(predicted_blur, target_blur)

    # Second part
    smooth = 1.
    lamb = 1.
    flat_pred = output.contiguous().view(-1) / scaling_factor
    flat_tar = target[:,1,:,:].contiguous().view(-1)

    A_sum = torch.sum(flat_tar * flat_pred)
    B_sum = torch.sum(flat_tar * flat_tar)

    dice_loss = 1 - ((2 * (flat_pred * flat_tar).sum() + smooth) / (A_sum + B_sum + smooth))

    return mse_location + lamb * dice_loss

def mse_ce_dice(output, target, scaling_factor):
    device = output.device
    bs = output.shape[0]

    psf_heatmap2 = matlab_style_gauss2D(shape=(7,7),sigma=1.0)
    gfilter2 = torch.from_numpy(psf_heatmap2).view(1, 1, 7, 7).to(device)

    predicted_blur = F.conv2d(output, gfilter2, stride=1, padding=3)

    target_mask = target[:,1,:,:].unsqueeze(1) 
    target_blur = F.conv2d(target_mask, gfilter2 * scaling_factor, stride=1, padding=3)

    mse_location = F.mse_loss(predicted_blur, target_blur)

    # Second part
    smooth = 1.
    lamb = 1.
    flat_pred = output.contiguous().view(-1) / scaling_factor
    flat_tar = target[:,1,:,:].contiguous().view(-1)

    A_sum = torch.sum(flat_tar * flat_pred)
    B_sum = torch.sum(flat_tar * flat_tar)

    dice_loss = 1 - ((2 * (flat_pred * flat_tar).sum() + smooth) / (A_sum + B_sum + smooth))

    return mse_location + lamb * dice_loss


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, alpha=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.alpha = alpha
        

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.alpha * self.target[index] + (1-self.alpha) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        return  final_loss

def matlab_style_gauss2D(shape,sigma):
    """ 
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma]) 
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2*sigma**2) )
    #h.astype(dtype=K.floatx())
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h*2.0
    maxV = h.max()
    h = h/maxV
    #print("max"+str(maxV))
    h = h.astype('float32')
    return h
    
def normalize_im(im, dmean, dstd):
    im = np.squeeze(im)
    im_norm = np.zeros(im.shape,dtype=np.float32)
    im_norm = (im - dmean)/dstd
    return im_norm

def Quickly_rotating_matrix_angleD_gamma_to_M(polar,azim,gamma):
    # transfer the angle from degree unit to the radial unit
    polar = polar/180*math.pi
    azim = azim/180*math.pi
    mux = torch.cos(azim) * torch.sin(polar)
    muy = torch.sin(azim) * torch.sin(polar)
    muz = torch.cos(polar)
    
    # size of muxx (pixel_size*pixel_size*frame_number)
    muxx = gamma * (mux ** 2) + (1.0 - gamma) / 3.0
    muyy = gamma * (muy ** 2) + (1.0 - gamma) / 3.0
    muzz = gamma * (muz ** 2) + (1.0 - gamma) / 3.0
    muxy = gamma * mux * muy
    muxz = gamma * mux * muz
    muyz = gamma * muz * muy    
    
    return (muxx, muyy, muzz, muxy, muxz, muyz)    
    
def SMLOM_Loss(spikes_pred, heatmap_true): # KL + MSE+L1
    # Expand the filter dimensions
    device = spikes_pred.device
    psf_heatmap = matlab_style_gauss2D(shape=(7,7),sigma=10.0)
    gfilter = torch.from_numpy(psf_heatmap).view(1, 1, 7, 7).to(device)


    psf_heatmap2 = matlab_style_gauss2D(shape=(7,7),sigma=1.0)
    gfilter2 = torch.from_numpy(psf_heatmap2).view(1, 1, 7, 7).to(device)

    gfilter3 = np.ones([7,7])
    gfilter3 = torch.from_numpy(gfilter3).view(1, 1, 7, 7).float().to(device)

    intensity_mask1 = heatmap_true[:,1,:,:].unsqueeze(1)   
    intensity_mask = F.conv2d(intensity_mask1, gfilter3, stride=1, padding=3)

    intensity_channel_est = spikes_pred[:,0,:,:]
    intensity_channel_est_sq1 = intensity_channel_est.unsqueeze(1) 
    intensity_est_blurred = F.conv2d(intensity_channel_est_sq1, gfilter2, stride=1, padding=3)
    
    theta_channel = spikes_pred[:,1,:,:]
    theta_channel_sq1 = theta_channel.unsqueeze(1)  
    theta_channel_sq = theta_channel_sq1 * intensity_mask
    
    phi_channel = spikes_pred[:,2,:,:]
    phi_channel_sq1 = phi_channel.unsqueeze(1) 
    phi_channel_sq = phi_channel_sq1 * intensity_mask

    gamma_channel = spikes_pred[:,3,:,:]
    gamma_channel_sq1 = gamma_channel.unsqueeze(1) 
    gamma_channel_sq = gamma_channel_sq1 * intensity_mask

    theta_true = heatmap_true[:,2,:,:].unsqueeze(1)
    theta_true_blurred = F.conv2d(theta_true, gfilter3, stride=1, padding=3)
    
    phi_true = heatmap_true[:,3,:,:].unsqueeze(1) 
    phi_true_blurred = F.conv2d(phi_true, gfilter3, stride=1, padding=3)
    
    gamma_true = heatmap_true[:,4,:,:].unsqueeze(1) 
    gamma_true_blurred = F.conv2d(gamma_true, gfilter3, stride=1, padding=3)
    
    intensity_true2 = heatmap_true[:,0,:,:].unsqueeze(1) 
    intensity_true2_blurred = F.conv2d(intensity_true2, gfilter2, stride=1, padding=3)
    
    
    muxx_theta, muyy_theta, muzz_theta, muxy_theta, muxz_theta, muyz_theta = Quickly_rotating_matrix_angleD_gamma_to_M(theta_channel_sq,phi_true_blurred,gamma_true_blurred)
    muxx_phi, muyy_phi, muzz_phi, muxy_phi, muxz_phi, muyz_phi = Quickly_rotating_matrix_angleD_gamma_to_M(theta_true_blurred,phi_channel_sq,gamma_true_blurred)
    muxx_gamma, muyy_gamma, muzz_gamma, muxy_gamma, muxz_gamma, muyz_gamma = Quickly_rotating_matrix_angleD_gamma_to_M(theta_true_blurred,phi_true_blurred,gamma_channel_sq)
    muxx_true_blurred, muyy_true_blurred, muzz_true_blurred, muxy_true_blurred, muxz_true_blurred, muyz_true_blurred = Quickly_rotating_matrix_angleD_gamma_to_M(theta_true_blurred,phi_true_blurred,gamma_true_blurred)
    # muxx_est, muyy_est, muzz_est, muxy_est, muxz_est, muyz_est = Quickly_rotating_matrix_angleD_gamma_to_M(theta_channel_sq1,phi_channel_sq1,gamma_channel_sq1)
    
    
    M_theta = torch.cat([muxx_theta, muyy_theta, muzz_theta, muxy_theta, muxz_theta, muyz_theta], 1)
    M_phi = torch.cat([muxx_phi, muyy_phi, muzz_phi, abs(muxy_phi), abs(muxz_phi), abs(muyz_phi),muxy_phi,muxz_phi, muyz_phi], 1)
    M_gamma = torch.cat([muxx_gamma, muyy_gamma, muzz_gamma,muxy_gamma, muxz_gamma, muyz_gamma], 1)
    M_true = torch.cat([muxx_true_blurred, muyy_true_blurred, muzz_true_blurred, muxy_true_blurred, muxz_true_blurred, muyz_true_blurred], 1)
    M_true_phi = torch.cat([muxx_true_blurred, muyy_true_blurred, muzz_true_blurred, abs(muxy_true_blurred), abs(muxz_true_blurred), abs(muyz_true_blurred),muxy_true_blurred, muxz_true_blurred, muyz_true_blurred], 1)
    

    
    #MSE 
    mse_M1_theta = F.mse_loss(M_theta*360, M_true*360)
    mse_M1_phi = F.mse_loss(M_phi*360, M_true_phi*360)
    mse_M1_gamma = F.mse_loss(M_gamma*360, M_true*360)
    #mse_M2 = losses.mean_squared_error(M_filtered,M_est)
    mse_intensity = F.mse_loss(intensity_true2_blurred/20, intensity_est_blurred/20)
    

    mse_theta2 = F.mse_loss(4*theta_channel_sq1, 4*theta_true_blurred)
    mse_phi2 = F.mse_loss(phi_channel_sq1, phi_true_blurred)
    mse_gamma2 = F.mse_loss(360*gamma_channel_sq1, 360*gamma_true_blurred)
    
    #L1 
    #psize = input_shape[0]
    #input_shape_mod = [1,psize,psize]
    l1_intensity = F.l1_loss(intensity_channel_est_sq1/20,torch.zeros(intensity_channel_est_sq1.shape).to(device))
    
    #loss = mse_theta*4+mse_phi+mse_gamma*(180**2)*10+ (mse_theta2*4+mse_phi2+mse_gamma2*(180**2))/1000+(mse_intensity+l1_intensity)/100
    loss = (mse_M1_theta+mse_M1_phi+mse_M1_gamma)+(mse_theta2+mse_phi2+mse_gamma2)/100+(mse_intensity+l1_intensity*2)*3
    #loss = mse_theta+mse_phi*4+mse_gamma*(2**2)+ (mse_theta2+mse_phi2*4+mse_gamma2*(2**2))/10000
# (mse_theta2+mse_phi2+mse_gamma2)/1000
    return loss, [mse_M1_theta.data.cpu().item(), mse_M1_phi.data.cpu().item(), mse_M1_gamma.data.cpu().item(), mse_intensity.data.cpu().item()]


def localization_loss(spikes_pred, heatmap_true, scaling_factor):
    device = spikes_pred.device

    psf_heatmap2 = matlab_style_gauss2D(shape=(7,7),sigma=1.0)
    gfilter2 = torch.from_numpy(psf_heatmap2).view(1, 1, 7, 7).to(device)

    intensity_channel_est = spikes_pred[:,0,:,:]
    intensity_channel_est_sq1 = intensity_channel_est.unsqueeze(1) 
    intensity_est_blurred = F.conv2d(intensity_channel_est_sq1, gfilter2, stride=1, padding=3)

    intensity_true2 = heatmap_true[:,0,:,:].unsqueeze(1) 
    intensity_true2_blurred = F.conv2d(intensity_true2, gfilter2, stride=1, padding=3)

    mse_intensity = F.mse_loss(intensity_est_blurred, intensity_true2_blurred)
    # mse_intensity = F.mse_loss(intensity_channel_est, intensity_true2)

    l1_intensity = F.l1_loss(intensity_channel_est,torch.zeros(intensity_channel_est.shape).to(device))

    total_loc_loss = mse_intensity + l1_intensity * 2

    return total_loc_loss
