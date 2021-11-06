# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:00:20 2019

@author: Hadi
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from pathlib import Path

HGG_dir = "G:/My Drive/Sem2/MachineLearning/Project/data/BraTS/BraTS18_small/training/HGG"
HGG_path = Path(HGG_dir)
image_types = ['flair', 't1', 't1ce', 't2'] # 1:flair, 2:t1, 3:t1ce, 4:t2
HGG_folder_list = os.listdir(HGG_path)
norms = []

class norm:
    def __init__(self, img_type, mean, std):
        self.img_type = img_type
        self.mean = mean
        self.std = std
        
for i in image_types:
    img_norm = norm(i,0.0,1.0)
    norms.append(img_norm)

type_number = 0
for i in image_types:
    data_temp_list = []
    for j in HGG_folder_list:
        img_path = os.path.join(HGG_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data() #img is a numpy array
        data_temp_list.append(img)
        
    print("pre_ image:", i, "____5" , "\n================")
    plt.imshow(data_temp_list[5][:,:,100])
    plt.ioff()
    plt.show()
   
    data_temp_list = np.asarray(data_temp_list)
    m = np.mean(data_temp_list) #105.57924098558448 for t2
    s = np.std(data_temp_list) #609.0591653806654 for t2
    norms[type_number].img_type = i
    norms[type_number].mean = m
    norms[type_number].std = s
    type_number += 1
    del data_temp_list
    
type_number = 0
for i in image_types:
    data_temp_list = []
    for j in HGG_folder_list:
        img_path = os.path.join(HGG_path, j, j + '_' + i + '.nii.gz')
        img = nib.load(img_path).get_data() #img is a numpy array
        img = (img - norms[type_number].mean)/norms[type_number].std
        data_temp_list.append(img)
        
    print("post_ image:", i, "____5" , "\n================")
    plt.imshow(data_temp_list[5][:,:,100])
    plt.ioff()
    plt.show()
    
    type_number += 1

    
print("iamges normalized: ", norms)























    