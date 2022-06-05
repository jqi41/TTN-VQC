#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:37:39 2022

@author: junqi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import h5py


def add_normal_noise(sig, target_snr_db=10):
    sig_watts = sig ** 2
    sig_avg_watts = sig_watts.mean()
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db 
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sig_watts))
    
    return sig + noise_volts


def add_laplace_noise(sig, target_snr_db=10):
    sig_watts = sig ** 2
    sig_avg_watts = sig_watts.mean()
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db 
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_volts = np.random.laplace(mean_noise, np.sqrt(noise_avg_watts), len(sig_watts))
    
    return sig + noise_volts



if __name__ == "__main__":
    batch_size_train = 60000
    batch_size_test = 10000
    learning_rate = 0.005
    snr_level = 15

    random_seed = 1
    torch.manual_seed(random_seed)
    SAVE_PATH = "dataset/MNIST/"

    train_loader = torch.utils.data.DataLoader(
                   torchvision.datasets.MNIST('dataset/', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                   torchvision.transforms.ToTensor(),
                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                  torchvision.datasets.MNIST('dataset/', train=False, download=True,
                  transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                  ])), batch_size=batch_size_test, shuffle=True)

    examples_train = enumerate(train_loader)
    examples_test  = enumerate(test_loader)
    batch_idx, (example_tr_data, example_tr_targets) = next(examples_train)
    batch_idx, (example_te_data, example_te_targets) = next(examples_test)
    
    example_tr_data = example_tr_data.reshape(-1, 784)
    example_te_data = example_te_data.reshape(-1, 784)

    dataset_tr_data3 = np.zeros((19138, 784))
    dataset_tr_data3_normal = np.zeros((19138, 784))
    dataset_tr_data3_laplace = np.zeros((19138, 784))
    dataset_te_data3 = np.zeros((3173, 784))
    dataset_te_data3_normal = np.zeros((3173, 784))
    dataset_te_data3_laplace = np.zeros((3173, 784))
    subset3 = [1, 3, 7]

    idx_tr_3 = 0
    idx_te_3 = 0
    
    for idx in range(len(example_tr_data)):
        if int(example_tr_targets[idx]) in subset3:
            dataset_tr_data3[idx_tr_3, :] = example_tr_data[idx, :]
            dataset_tr_data3_normal[idx_tr_3, :] = add_normal_noise(example_tr_data[idx, :], snr_level)
            dataset_tr_data3_laplace[idx_tr_3, :] = add_laplace_noise(example_tr_data[idx, :], snr_level)
            idx_tr_3 += 1

    for idx in range(len(example_te_data)):
        if int(example_te_targets[idx]) in subset3:
            dataset_te_data3[idx_te_3, :] = example_te_data[idx, :] 
            dataset_te_data3_normal[idx_te_3, :] = add_normal_noise(example_te_data[idx, :], snr_level)
            dataset_te_data3_laplace[idx_te_3, :] = add_laplace_noise(example_te_data[idx, :], snr_level)
            idx_te_3 += 1
            
    hf = h5py.File("dataset/data_sub_noise_mnist.h5", 'w')
    hf.create_dataset('tr_data3', data=dataset_tr_data3)
    hf.create_dataset('tr_data3_normal', data=dataset_tr_data3_normal)
    hf.create_dataset("tr_data3_laplace", data=dataset_tr_data3_laplace)
    hf.create_dataset('te_data3', data=dataset_te_data3)
    hf.create_dataset('te_data3_normal', data=dataset_te_data3_normal)
    hf.create_dataset("te_data3_laplace", data=dataset_te_data3_laplace)  
        
    
    
    
    
    
