from cgi import test
from math import cos
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time
from models.models import ConvLSTM,PhyCell, EncoderRNN
from constrain_moments import K2M
from skimage.metrics import structural_similarity as ssim
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.cm as cm
from random import randrange
import matplotlib.animation as animation

import phydnet_predict

# Path
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_image = dir_path + "/loader_image"

train_loader_file = dir_path_image + "/train_loader_1"
test_loader_file = dir_path_image + "/test_loader_1"
print(dir_path)

# Loaders
batch_size = 32
n_frames= 10
costmap_size = 60
samples_len = 12546 - 2*n_frames
img_channels = 3

train_len = int(samples_len*0.8)
test_len = int(samples_len*0.2)

train_loader_len = int(train_len/10)
test_loader_len = int(test_len/10)

train_test_sequence = np.zeros([train_len + test_len, n_frames, costmap_size, costmap_size])
train_sequence = np.zeros([train_len, n_frames, costmap_size, costmap_size])
test_sequence = np.zeros([test_len, n_frames, costmap_size, costmap_size])

train_loader = np.zeros([train_loader_len, batch_size , img_channels, costmap_size+4, costmap_size+4])
test_loader = np.zeros([test_loader_len, batch_size , img_channels, costmap_size+4, costmap_size+4])

# Module

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_path_checkpoint = dir_path + '/best_networks/encoder_phydnet_128_11.pth'


def main():

    # Open data. Numpy arrays
    data = np.zeros([1,60,60])
    for i in range(1,6,1):
        # Concatenate data
        with open(dir_path_image + "/costmap" + str(i) + ".npy", 'rb') as f:
            temp_data = np.load(f)
        data = np.concatenate((data, temp_data), axis=0)

    for itr in range(train_len + test_len):
        train_test_sequence[itr,:,:,:] = data[itr: itr + n_frames,:,:]
    
    # Shuffle
    np.random.shuffle(train_test_sequence)
    
    # Separate train and test
    for itr in range(train_len):
        train_sequence[itr,:,:,:] = train_test_sequence[itr,:,:,:]

    for itr in range(test_len):
        test_sequence[itr,:,:,:] = train_test_sequence[itr + train_len,:,:,:]

    # Prediction module
    prediction_module =  phydnet_predict.PhydNet(dir_path_checkpoint)

    # Generate images
    for itr in range(train_loader_len):
        video = np.zeros([batch_size,n_frames,1,costmap_size,costmap_size])
        for jtr in range(batch_size):
            rand = randrange(train_len)
            video[jtr,:,0,:,:] = train_sequence[rand,:,:,:]
        predictions = prediction_module.predict(video)
        for i in range(img_channels):
            train_loader[itr,:,i,0:costmap_size,0:costmap_size] = np.squeeze(predictions[:,-1,:,:])

    for itr in range(test_loader_len):
        video = np.zeros([batch_size,n_frames,1,costmap_size,costmap_size])
        for jtr in range(batch_size):
            rand = randrange(test_len)
            video[jtr,:,0,:,:] = test_sequence[rand,:,:,:]
        predictions = prediction_module.predict(video)
        for i in range(img_channels):
            test_loader[itr,:,i,0:costmap_size,0:costmap_size] = np.squeeze(predictions[:,-1,:,:])
    
    plt.figure()
    imgplot = plt.imshow(np.squeeze(train_loader[0,0,0,:,:]))
    plt.show()

    plt.figure()
    imgplot = plt.imshow(np.squeeze(test_loader[0,0,0,:,:]))
    plt.show()

    with open(train_loader_file, 'wb') as f:
        np.save(f, train_loader)
    f.close()

    with open(test_loader_file, 'wb') as f:
        np.save(f, test_loader)
    f.close()


if __name__ == '__main__':
    main()