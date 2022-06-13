import sys

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data.dataset import Subset
import torch
import torch.nn.functional as F
import random 
import json
import os
from utils import np_transforms
import h5py
from scipy.io import loadmat

def get_mat(mat_obj):
    kk = list(mat_obj.keys())
    return mat_obj[kk[-1]]

def get_microscopy(root, cfg_trainer, val = False):
    if val == False:
        matfile_XY = h5py.File(root+cfg_trainer['filename_xy'], 'r')
        matfile_intensity = h5py.File(root+cfg_trainer['filename_intensity'], 'r')
        matfile_theta = h5py.File(root+cfg_trainer['filename_theta'], 'r')
        matfile_phi = h5py.File(root+cfg_trainer['filename_phi'], 'r')
        matfile_omega = h5py.File(root+cfg_trainer['filename_omega'], 'r')

        Intensity_channel = np.transpose(np.array(matfile_intensity['image_intensity'],dtype='float32'))
        Theta_channel = np.transpose(np.array(matfile_theta['image_theta'],dtype='float32'))
        Phi_channel = np.transpose(np.array(matfile_phi['image_phi'],dtype='float32'))
        Omega_channel = np.transpose(np.array(matfile_omega['image_gamma'],dtype='float32'))

        XY_channel = np.transpose(np.array(matfile_XY['image_raw_xy']))
    else:
        matfile_X = get_mat(loadmat(root+cfg_trainer['filename_x']))
        matfile_Y = get_mat(loadmat(root+cfg_trainer['filename_y']))
        matfile_intensity = get_mat(loadmat(root+cfg_trainer['filename_intensity']))
        matfile_theta = get_mat(loadmat(root+cfg_trainer['filename_theta']))
        matfile_phi = get_mat(loadmat(root+cfg_trainer['filename_phi']))
        matfile_omega = get_mat(loadmat(root+cfg_trainer['filename_omega']))

        Intensity_channel = np.array(matfile_intensity,dtype='float32')[:,:,::2]
        Theta_channel = np.array(matfile_theta,dtype='float32')[:,:,::2]
        Phi_channel = np.array(matfile_phi,dtype='float32')[:,:,::2]
        Omega_channel = np.array(matfile_omega,dtype='float32')[:,:,::2]
        XY_channel = np.array(np.stack([matfile_X, matfile_Y], 0),dtype='float32')[:,:,:,::2]

    Input_channel = XY_channel.transpose(3,0,1,2)
    Intensity_channel_t = np.expand_dims(Intensity_channel,axis=-1)
    Intensity_channel_t = Intensity_channel_t.transpose(2,3,0,1)
    Theta_channel_t = np.expand_dims(Theta_channel,axis=-1)
    Theta_channel_t = Theta_channel_t.transpose(2,3,0,1)
    Phi_channel_t = np.expand_dims(Phi_channel,axis=-1)
    Phi_channel_t = Phi_channel_t.transpose(2,3,0,1)
    #Phi_channel_t = np.remainder(Phi_channel_t,180.01)
    Omega_channel_t = np.expand_dims(Omega_channel,axis=-1)
    Omega_channel_t = Omega_channel_t.transpose(2,3,0,1)

    Intensity_channel_mask = np.array(Intensity_channel_t,copy=True)
    Intensity_channel_mask[Intensity_channel_mask>1]=1   
               #Output_channel=np.concatenate((Output_channel_i,Intensity_channel_t,Theta_channel_t,Phi_channel_t,Gamma_channel_t),axis=-1)
    Output_channel=np.concatenate((Intensity_channel_t,Intensity_channel_mask,Theta_channel_t,Phi_channel_t,Omega_channel_t),axis=1)
    print("Input Channel Shape => " + str(Input_channel.shape))
    print("Output Channel Shape => " + str(Output_channel.shape))

    Intensity_channel_t=np.array([])
    Theta_channel_t=np.array([])
    Phi_channel_t=np.array([])
    Omega_channel_t=np.array([])
    XY_channel=np.array([])
    # XY_channel_denoised=np.array([])
    Theta_channel=np.array([])
    Phi_channel=np.array([])
    Omega_channel=np.array([])
    Intensity_channel_mask=np.array([])

    if val == False:
        train_idxs, val_idxs = train_val_split(Output_channel, cfg_trainer['subset_percent'])
        train_idxs = np.sort(train_idxs)
        val_idxs = np.sort(val_idxs)

        train_dataset = microscopy(root, cfg_trainer, train_idxs, Input_channel, Output_channel)
        val_dataset = microscopy(root, cfg_trainer, val_idxs, Input_channel, Output_channel,val=True)
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}") 
        return train_dataset, val_dataset
    else:
        idxs = np.arange(len(Output_channel))
        val_dataset = microscopy(root, cfg_trainer, idxs, Input_channel, Output_channel, val)
        print(f"Validation: {len(val_dataset)}") 
        return val_dataset
    


def train_val_split(base_dataset, subset_percent = 1.0):
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * subset_percent * 0.9)
    val_n = int(len(base_dataset) * 0.9)
    train_idxs = []
    val_idxs = []

    idxs = np.arange(len(base_dataset))
    np.random.shuffle(idxs)
    train_idxs.extend(idxs[:train_n])
    val_idxs.extend(idxs[val_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class microscopy(torch.utils.data.Dataset):
    def __init__(self, root, cfg_trainer, indexs, data, labels, val=False):
        self.data = data[indexs]
        self.labels = labels[indexs]
        if val:
            self.trf = np_transforms.Compose([
                        np_transforms.ToTensor()
                    ])
        else:
            self.trf = np_transforms.Compose([
                            #np_transforms.Scale(size=(256, 256)),
                            #np_transforms.RandomCrop(size=(128, 128)),
                            #np_transforms.RandomVerticalFlip(prob=0.5),
                            #np_transforms.RandomHorizontalFlip(prob=0.5),
                            #np_transforms.RotateImage(angles=(-15, 15)),
                            np_transforms.ToTensor(),
                        ])


            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)


        img = self.transform(img)
        img_channel = img.shape[0]
        target_channel = target.shape[0]
        
        img_target = np.concatenate((img,target),axis=0)
        
        img_target = self.trf(np.transpose(img_target, [1,2,0]))#.permute(2,1,0)

        img = img_target[:img_channel,...]
        target = img_target[img_channel:img_channel+target_channel,...]

        #target = torch.tensor(target).float()
        return img, target


    def __len__(self):
        return len(self.data)

    def transform(self,data):
        mean = np.array([2.9542, 2.9550]).reshape([2,1,1])
        std = np.array([2.4386, 2.4343]).reshape([2,1,1])
        return torch.tensor((data - mean)/std).float()

