# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:38:38 2019

@author: Hadi
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os, csv, pickle, gc, sys
from pathlib import Path
from matplotlib.animation import FuncAnimation

import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56} , gpu_options=gpu_options))

import keras
keras.backend.set_session(sess)
from tensorflow.python.client import device_lib
from keras import optimizers
from keras.layers import Input, concatenate, Lambda 
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers import  BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.losses import binary_crossentropy
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
K.set_session(sess)


'''
===================================================
initializations
===================================================
'''
#HGG_dir = "data/BraTS/BraTS17/HGG"		
#LGG_dir = "data/BraTS/BraTS17/LGG"		
#survival_dir = "data/BraTS/BraTS17/survival_data.csv"

HGG_dir = "G:/My Drive/Sem2/MachineLearning/Project/data/BraTS/BraTS18/training/HGG"
LGG_dir = "G:/My Drive/Sem2/MachineLearning/Project/data/BraTS/BraTS18/training/LGG"
survival_dir = "G:/My Drive/Sem2/MachineLearning/Project/data/BraTS/BraTS18/training/survival_data.csv"

save_dir = "saved_data/"

if not os.path.exists(save_dir): os.makedirs(save_dir)
save_normal_images_dir = "normal_images/"

data_preprocess = 1 # 1 to perform data preparation, 0 if already perpared
load_the_model  = 0 # 0 to train 
evaluate_model = 1 # 0 to avoid evaluation
plot_results = 1 #0 to avoid plotting
load_norms = 0 # 0 to calculate mean and std

#define portion of train data
HGG_train_percent = 70
LGG_train_percent = 70

#network parameters
batch_size = 4
learning_rate = 0.001 
lr_decay = 0.5
beta1 = 0.9
epochs = 5000

img_height = 240
img_width = 240
img_channels = 4

'''
===================================================
general functions
===================================================
'''
class norm:
    def __init__(self, img_type, mean, std):
        self.img_type = img_type
        self.mean = mean
        self.std = std

image_types = ['flair', 't1', 't1ce', 't2'] # 1:flair, 2:t1, 3:t1ce, 4:t2
norms = []
    
def jaccard(ground_truth, y_pred, smooth=100):
    intersection = K.sum(K.abs(ground_truth * y_pred), axis=-1)
    union = K.sum(K.abs(ground_truth) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (union - intersection + smooth)
    return (1 - jac) * smooth  

def dice(ground_truth, y_pred):
    smooth = 1
    ground_truth_f = K.flatten(ground_truth)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(ground_truth_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(ground_truth_f) + K.sum(y_pred_f) + smooth)

def dice_loss(ground_truth, y_pred):
    return 0.5 * binary_crossentropy(ground_truth, y_pred) - dice(ground_truth, y_pred)

def load_img(data_path, counter, image_type):
    img_path = os.path.join(data_path, counter, counter + '_' + image_type + '.nii.gz')
    img = nib.load(img_path).get_data()
    return img


'''
===================================================
visualization functions
===================================================
'''

def show_image(nifti_img, slice_n):
    plt.ion()
    #img = nib.load(dir + 'c0001s0004t01.nii.gz')
    img = nib.load(nifti_img)
    img1_arr = img.get_fdata()
    plt.figure(figsize=(6,6))
    print(nifti_img.shape)
    plt.imshow(img1_arr[:,slice_n,:])
    plt.ioff()
    plt.show()

def show_mri_images(X, y, preds, binary_preds, ix=None):
    for ix in range(ix,ix+155):
        print ('=============== image({:d}) ================'.format(ix))
        fig, ax = plt.subplots(1, 8, figsize=(20, 10))
        ax[0].imshow(X[ix,:,:,0], cmap='seismic')
        ax[0].set_title('flair')

        ax[1].imshow(X[ix,:,:,1], cmap='seismic')
        ax[1].set_title('t1')
        
        ax[2].imshow(X[ix,:,:,2], cmap='seismic')
        ax[2].set_title('t1ce')
        
        ax[3].imshow(X[ix,:,:,3], cmap='seismic')
        ax[3].set_title('t2')

        ax[4].imshow(y[ix,:,:,0].squeeze(), vmin=0, vmax=1)
        ax[4].set_title('seg')
        
        ax[5].imshow(preds[ix,:,:,0].squeeze(), vmin=0, vmax=1)
        ax[5].set_title('Predicted')

        ax[6].imshow(binary_preds[ix,:,:,0].squeeze(), vmin=0, vmax=1)
        ax[6].set_title('Predicted binary')
        
        plt.savefig(save_dir + 'plots/plot_' + str(ix) + '.png', format='png')
        

fig1, ax1 = plt.subplots(1, 8, figsize=(20, 10))
def show_gif(i, X, y, preds, binary_preds):
    label = 'image {0}'.format(i)
    print(label)
    
    ax1[0].imshow(X[i,:,:,0], cmap='seismic')
    ax1[0].set_title('flair')

    ax1[1].imshow(X[i,:,:,1], cmap='seismic')
    ax1[1].set_title('t1')
    
    ax1[2].imshow(X[i,:,:,2], cmap='seismic')
    ax1[2].set_title('t1ce')
    
    ax1[3].imshow(X[i,:,:,3], cmap='seismic')
    ax1[3].set_title('t2')

    ax1[4].imshow(y[i,:,:,0], vmin=0, vmax=1)
    ax1[4].set_title('seg')
    
    ax1[5].imshow(preds[i,:,:,0], vmin=0, vmax=1)
    ax1[5].set_title('Predicted')

    ax1[6].imshow(binary_preds[i,:,:,0], vmin=0, vmax=1)
    ax1[6].set_title('Predicted binary')
    
    ax1.set_xlabel(label)
    plt.show()
    return  ax1

def save_gif(X, y, preds, binary_preds, ix=None):
    anim = FuncAnimation(fig1, show_gif, frames=np.arange(0, ix), interval=5)
    #anim.save(save_dir + 'predictions.gif', dpi=80, writer='imagemagick')
    #anim.save(save_dir + 'predictions.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def show_chart(trained_model):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 8))
    plt.xlim(0, 10000)
    plt.ylim(0, 1)
    plt.plot(trained_model.history["loss"], label="train_loss")
    plt.plot(trained_model.history["val_loss"], label="val_loss")
    plt.plot(trained_model.history["dice"], label="dice")
    plt.plot(trained_model.history["val_dice"], label="val_dice")
    plt.plot(trained_model.history["acc"], label="train_acc")
    plt.plot(trained_model.history["val_acc"], label="val_acc")
    plt.plot(trained_model.history["jaccard"], label="jaccard")
    plt.plot(trained_model.history["val_jaccard"], label="val_jaccard")
    plt.plot( np.argmin(trained_model.history["val_loss"]), np.min(trained_model.history["val_loss"]), marker="x", color="r", label="best model")
    plt.title("training charts")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("train_chart.png")
    
'''
===================================================
file functions
===================================================
'''
    
def save_pickle(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj, protocol=4)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def load_pickle(filepath):
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj


'''
===================================================
define unet model
===================================================
''' 
def unet_model(input_img, activation_type = 'elu', filters=16, dropout=0.5, batch_normalize=True):  
    
    # encoder
    c1 = Conv2D(filters = filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c1')(input_img)
    if batch_normalize: c1 = BatchNormalization()(c1)
    c1 = Activation(activation_type)(c1)
    c1 = Conv2D(filters = filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c1_2')(c1)
    if batch_normalize: c1 = BatchNormalization()(c1)
    c1 = Activation(activation_type)(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = Conv2D(filters = 2*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c2')(p1)
    if batch_normalize: c2 = BatchNormalization()(c2)
    c2 = Activation(activation_type)(c2)
    c2 = Conv2D(filters = 2*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c2_2')(c2)
    if batch_normalize: c2 = BatchNormalization()(c2)
    c2 = Activation(activation_type)(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = Conv2D(filters = 4*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c3')(p2)
    if batch_normalize: c3 = BatchNormalization()(c3)
    c3 = Activation(activation_type)(c3)
    c3 = Conv2D(filters = 4*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c3_2')(c3)
    if batch_normalize: c3 = BatchNormalization()(c3)
    c3 = Activation(activation_type)(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = Conv2D(filters = 8*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c4')(p3)
    if batch_normalize: c4 = BatchNormalization()(c4)
    c4 = Activation(activation_type)(c4)
    c4 = Conv2D(filters = 8*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c4_2')(c4)
    if batch_normalize: c4 = BatchNormalization()(c4)
    c4 = Activation(activation_type)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = Conv2D(filters = 8*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c5')(p4)
    if batch_normalize: c5 = BatchNormalization()(c5)
    c5 = Activation(activation_type)(c5)
    c5 = Conv2D(filters = 8*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c5_2')(c5)
    if batch_normalize: c5 = BatchNormalization()(c5)
    c5 = Activation(activation_type)(c5)
    
    # decoder
    u6 = Conv2DTranspose(8*filters, (3, 3), strides=(2, 2), padding='same', name='u6') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = Conv2D(filters = 8*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c6')(u6)
    if batch_normalize: c6 = BatchNormalization()(c6)
    c6 = Activation(activation_type)(c6)
    c6 = Conv2D(filters = 8*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c6_2')(c6)
    if batch_normalize: c6 = BatchNormalization()(c6)
    c6 = Activation(activation_type)(c6)

    u7 = Conv2DTranspose(4*filters, (3, 3), strides=(2, 2), padding='same', name = 'u7') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = Conv2D(filters = 4*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c7')(u7)
    if batch_normalize: c7 = BatchNormalization()(c7)
    c7 = Activation(activation_type)(c7)
    c7 = Conv2D(filters = 4*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c7_2')(c7)
    if batch_normalize: c7 = BatchNormalization()(c7)
    c7 = Activation(activation_type)(c7)

    u8 = Conv2DTranspose(2*filters,kernel_size = (3, 3), strides=(2, 2), padding='same', name = 'u8') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = Conv2D(filters = 2*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c8')(u8)
    if batch_normalize: c8 = BatchNormalization()(c8)
    c8 = Activation(activation_type)(c8)
    c8 = Conv2D(filters = 2*filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c8_2')(c8)
    if batch_normalize: c8 = BatchNormalization()(c8)
    c8 = Activation(activation_type)(c8)

    u9 = Conv2DTranspose(filters, kernel_size =(3, 3), strides=(2, 2), padding='same', name = 'u9') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = Conv2D(filters = filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name='c9')(u9)
    if batch_normalize: c9 = BatchNormalization()(c9)
    c9 = Activation(activation_type)(c9)
    c9 = Conv2D(filters = filters, kernel_size = (3, 3) ,kernel_initializer = 'he_normal', padding = 'same', name ='c9_2')(c9)
    if batch_normalize: c9 = BatchNormalization()(c9)
    c9 = Activation(activation_type)(c9)
    
    outputs = Conv2D(1, kernel_size =(1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    
    '''plot_model(model, to_file=os.path.join(save_dir +"model.png"))
    if os.path.exists(os.path.join(save_dir +"model.txt")):
        os.remove(os.path.join(save_dir +"model.txt"))
    with open(os.path.join(save_dir +"model.txt"),'w') as fh:
        model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))'''
        
    return model


'''
===================================================
data preprocessing
===================================================
'''

print(device_lib.list_local_devices())

if data_preprocess == 1:
    HGG_path = Path(HGG_dir)
    LGG_path = Path(LGG_dir)
    
    for i in image_types:
        img_norm = norm(i,0.0,1.0)
        print(img_norm)
        norms.append(img_norm)
    
    HGG_path = Path(HGG_dir)
    LGG_path = Path(LGG_dir)
    
    hgg_folder_list = os.listdir(HGG_path)
    lgg_folder_list = os.listdir(LGG_path)
    
    num_of_HGGs=len(hgg_folder_list)
    num_of_LGGs=len(lgg_folder_list)
    total_folders = num_of_HGGs + num_of_LGGs
    
    print("There are", num_of_HGGs , "HGG folders, and", num_of_LGGs , "LGG folders.")
    
    id_list = []
    hgg_id_list = []
    lgg_id_list = []
    age_list =[]
    period_list = []
    
    with open(survival_dir, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for idx, record in enumerate(reader): id_list.append(record[0])
            
    for i in id_list:
        if i in hgg_folder_list: hgg_id_list.append(i)
        elif i in lgg_folder_list: lgg_id_list.append(i)
    
    print("There are", len(id_list), "survival cases.")
    print("There are", len(hgg_id_list), "HGG survivals, and", len(lgg_id_list), "LGG survivals.")
    
    hgg_split_index = list(range(0, len(hgg_id_list)))
    lgg_split_index = list(range(0, len(lgg_folder_list)))
    
    hgg_train_split = int((100-HGG_train_percent)*num_of_HGGs/100)
    lgg_train_split = int((100-LGG_train_percent)*num_of_LGGs/100)
    hgg_test_split = int(HGG_train_percent*num_of_HGGs/100)
    lgg_test_split = int(LGG_train_percent*num_of_LGGs/100)
    
    hgg_train_split_index = hgg_split_index[:-hgg_train_split]
    hgg_split_test_index = hgg_split_index[-hgg_train_split:]
    
    lgg_train_split_index = lgg_split_index[:-lgg_train_split]
    lgg_split_test_index = lgg_split_index[-lgg_train_split:]
    
    hgg_test_id_list = [hgg_id_list[i] for i in hgg_split_test_index]
    hgg_train_id_list = [hgg_id_list[i] for i in hgg_train_split_index]
    
    lgg_test_id_list = [lgg_folder_list[i] for i in lgg_split_test_index]
    lgg_train_id_list = [lgg_folder_list[i] for i in lgg_train_split_index]
    
    
    #================================== collect train and test images
    
    train_output_filepath = save_dir + 'train_output.pickle'
    train_target_filepath = save_dir + 'train_target.pickle'
    test_input_filepath = save_dir + 'test_input.pickle'
    test_output_filepath = save_dir + 'test_output.pickle'
    
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    
    '''
    mris[0] -> flair
    mris[1] -> t1
    mris[2] -> t1ce
    mris[3] -> t2
    '''
    print("******* collecting train data from HGG")
    for i in hgg_train_id_list:
        mris = []
        type_number = 0
        seg_img = load_img(HGG_path, i, 'seg')
        seg_img = np.transpose(seg_img, (1, 0, 2))
        for j in image_types:
            img = load_img(HGG_path, i, j)
            type_number += 1
            img = img.astype(np.float32)
            mris.append(img)

        for j in range(mris[0].shape[2]):
            all_mris = np.stack((mris[0][:, :, j], mris[1][:, :, j], mris[2][:, :, j], mris[3][:, :, j]), axis=2)
            all_mris = np.transpose(all_mris, (1, 0, 2))
            all_mris.astype(np.float32)
            train_input.append(all_mris)
    
            seg_2d = seg_img[:, :, j]
            seg_2d.astype(int)
            train_output.append(seg_2d)
        del mris
        gc.collect()
    
    print("******* collecting train data from LGG")
    for i in lgg_train_id_list:
        mris = []
        type_number = 0
        seg_img = load_img(LGG_path, i, 'seg')
        seg_img = np.transpose(seg_img, (1, 0, 2))
        for j in image_types:
            img = load_img(LGG_path, i, j)
            type_number += 1
            img = img.astype(np.float32)
            mris.append(img)
            
        for j in range(mris[0].shape[2]):
            all_mris = np.stack((mris[0][:, :, j], mris[1][:, :, j], mris[2][:, :, j], mris[3][:, :, j]), axis=2)
            all_mris = np.transpose(all_mris, (1, 0, 2))
            all_mris.astype(np.float32)
            train_input.append(all_mris)
    
            seg_2d = seg_img[:, :, j]
            seg_2d.astype(int)
            train_output.append(seg_2d)
        del mris
        gc.collect()
    
    print("******* collecting test data from HGG")
    for i in hgg_test_id_list:
        mris = []
        type_number = 0
        seg_img = load_img(HGG_path, i, 'seg')
        seg_img = np.transpose(seg_img, (1, 0, 2))
        
        for j in image_types:
            img = load_img(HGG_path, i, j)
            type_number += 1
            img = img.astype(np.float32)
            mris.append(img)
 
        for j in range(mris[0].shape[2]): # 2 is slice number
            all_mris = np.stack((mris[0][:, :, j], mris[1][:, :, j], mris[2][:, :, j], mris[3][:, :, j]), axis=2)
            all_mris = np.transpose(all_mris, (1, 0, 2))
            all_mris.astype(np.float32)
            #nib.save(all_mris, os.path.join(save_normal_images_dir, HGG_path, i, i + '_' + j + '.nii.gz'))
            test_input.append(all_mris)
    
            seg_2d = seg_img[:, :, j]
            seg_2d.astype(int)
            test_output.append(seg_2d)
        del mris
        gc.collect()
    
    print("******* collecting test data from LGG")
    for i in lgg_test_id_list:
        mris = []
        type_number = 0
        seg_img = load_img(LGG_path, i, 'seg')
        seg_img = np.transpose(seg_img, (1, 0, 2))
        for j in image_types:
            img = load_img(LGG_path, i, j)
            type_number += 1
            img = img.astype(np.float32)
            mris.append(img)
        for j in range(mris[0].shape[2]):
            all_mris = np.stack((mris[0][:, :, j], mris[1][:, :, j], mris[2][:, :, j], mris[3][:, :, j]), axis=2)
            all_mris = np.transpose(all_mris, (1, 0, 2))
            all_mris.astype(np.float32)
            test_input.append(all_mris)
    
            seg_2d = seg_img[:, :, j]
            seg_2d.astype(int)
            test_output.append(seg_2d)
        del mris
        gc.collect()
    
    train_input = np.asarray(train_input, dtype=np.float32)
    train_output = np.asarray(train_output, dtype=np.float32)
    test_input = np.asarray(test_input, dtype=np.float32)
    test_output = np.asarray(test_output, dtype=np.float32)
    
    try:
        X_train = train_input
        Y_train = train_output[:,:,:,np.newaxis]
        X_test = test_input
        Y_test = test_output[:,:,:,np.newaxis]
    except Exception:
        pass
    
    #save_pickle(X_train, train_output_filepath)
    #save_pickle(Y_train, train_target_filepath)
    #save_pickle(X_test, test_input_filepath)
    #save_pickle(Y_test, test_output_filepath)
    del train_input,train_output, test_input, test_output
    
    print("finished with data preparation.")
      

def generate_data(X_data, Y_data, batch_size):
    
    samples_per_epoch = total_folders
    number_of_batches = samples_per_epoch/batch_size
    counter=0
        
    while True:
        
        X_batch = X_data[batch_size*counter:batch_size*(counter+1)]
        Y_batch = Y_data[batch_size*counter:batch_size*(counter+1)]
        
        counter += 1
        
        yield X_batch, Y_batch
        
        if counter >= number_of_batches:
            counter = 0
    

'''
===================================================
main function
===================================================
'''

callbacks = [
    CSVLogger(save_dir + "train_log.csv"),
    EarlyStopping(patience=1000000, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(save_dir + 'model.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

if __name__ == "__main__":
    
    adam = optimizers.Adam(lr=learning_rate, beta_1=beta1, decay=lr_decay, amsgrad=False)
    input_img = Input((img_height, img_width, img_channels), name='img')
    model = unet_model(input_img = input_img, activation_type = 'relu', filters=16, dropout=0.5, batch_normalize=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy",dice,jaccard])
    #model.compile(optimizer=Adam(), loss = dice_loss, metrics= ["accuracy", dice, jaccard])
    #model.compile(optimizer=adam(), loss = dice_loss, metrics= ["accuracy", dice, jaccard])
    model.summary()
    print("\n*************** model created.")
    
    if data_preprocess==0:
        X_train = load_pickle(save_dir + 'train_output.pickle')
        Y_train = load_pickle(save_dir + 'train_target.pickle')
        X_test = load_pickle(save_dir + 'test_input.pickle')
        Y_test = load_pickle(save_dir + 'test_output.pickle')
        
        if load_the_model == 0:
            H = model.fit(X_train, Y_train, 
                  batch_size=batch_size, 
                  epochs = epochs, 
                  callbacks=callbacks,
                  steps_per_epoch = total_folders/batch_size, 
                  validation_data=(X_test, Y_test),
                  validation_steps= total_folders/batch_size*2)
            print("\n*************** training done.\n") 
            show_chart(H)
        else:
            H = model.load_weights(save_dir + 'model.10-0.03.h5')
            print("\n*************** model loaded.")
            #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            #model.compile(optimizer=adam, loss=dice_loss, metrics=['accuracy',dice,jaccard])
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy',dice,jaccard])
            if evaluate_model == 1:
                print("\n**************** evaluating the model:\n")
                score = model.evaluate(X_test, Y_test, verbose=1)
                print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
                print("\n**************** evaluating done\n")
    
    else:
        if load_the_model == 0:
            H = model.fit_generator(generate_data(X_train,Y_train,batch_size), 
                        epochs= epochs,
                        steps_per_epoch = total_folders/batch_size, 
                        validation_data=generate_data(X_test,Y_test,batch_size*2),
                        callbacks=callbacks,
                        validation_steps= total_folders/batch_size*2)
            print("\n***************\ntraining done.\n") 
            show_chart(H)
        else:
            H = model.load_weights(save_dir + 'model.10-0.03.h5')
            print("\n*************** model loaded.\n")
            #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            #model.compile(optimizer=adam, loss=dice_loss, metrics=['accuracy',dice,jaccard])
            #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy',dice,jaccard])
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy',dice,jaccard])
            if evaluate_model == 1:
                print("\n**************** evaluating the model:\n")
                score = model.evaluate(X_test, Y_test, verbose=1)
                print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
                print("\n**************** evaluating done\n")
    

    if plot_results == 1:
        print("\npredicting: ****************\n")

        preds_train = model.predict(X_train, verbose=1)
        preds_val = model.predict(X_test, verbose=1)
        
        # Threshold predictions
        preds_train_t = (preds_train > 0.5).astype(np.uint8)
        preds_val_t = (preds_val > 0.5).astype(np.uint8)
        
        show_mri_images(X_train, Y_train, preds_train, preds_train_t, ix=970)
        #save_gif(X_train, Y_train, preds_train, preds_train_t, ix=100)
        
        show_mri_images(X_test, Y_test, preds_val, preds_val_t, ix=80)
        #save_gif(X_test, Y_test, preds_val, preds_val_t, ix=100)
        
    K.clear_session()
    del model
    del H
        

