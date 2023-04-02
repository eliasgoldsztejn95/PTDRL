#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray
import numpy as np 
import os 
import yaml
from yaml.loader import SafeLoader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.animation as animation
import cv2
import matplotlib.cm as cm
from random import randrange
import matplotlib.animation as animation

################################################################################################################
# Open costmap.npy and write train_loader and test_loader. Their
# shape is: [train_loader_len/test_loader_len, batch_size, img_channels, costmap_size, costmap_size]
# This is the image loader.
################################################################################################################

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.split("/")
dir_path = dir_path[:-2]
dir_path += ["recordings"]
dir_path = '/'.join(dir_path)
dir_path += "/costmap_image"

file = "costmap"
np_file = "/" + file + ".npy"

dir_path_np = dir_path

dir_path_np += np_file

train_loader_file = dir_path
train_loader_file += "/train_loader_2.npy"

test_loader_file = dir_path
test_loader_file += "/test_loader_2.npy"

batch_size = 32
costmap_size = 64
samples_len = 4946
num_files = 2

train_len = int(samples_len*0.8) # 80% train, 20% test
test_len = int(samples_len*0.2)

train_loader_len = int(train_len/10)
test_loader_len = int(test_len/10)

img_channels = 3

train_sequence = np.zeros([train_len, costmap_size, costmap_size])
test_sequence = np.zeros([test_len, costmap_size, costmap_size])

train_loader = np.zeros([train_loader_len, batch_size,img_channels, costmap_size, costmap_size])
test_loader = np.zeros([test_loader_len, batch_size,img_channels, costmap_size, costmap_size])


def main():

    data = np.zeros([1,60,60])
    for i in range(1,num_files,1):
        # Concatenate data
        with open(dir_path + "/costmap_inflation_600_" + str(i) + ".npy", 'rb') as f:
            temp_data = np.load(f)
        data = np.concatenate((data, temp_data), axis=0)
    
    print(data.shape)
    print(train_len)
    
    # Shuffle data
    np.random.shuffle(data)

    for itr in range(train_len):
        train_sequence[itr,0:60,0:60] = data[itr,:,:]/100

    for itr in range(test_len):
        test_sequence[itr,0:60,0:60] = data[itr + train_len,:,:]/100
    print(itr + train_len)

    
    for itr in range(train_loader_len):
        for jtr in range(batch_size):
            rand = randrange(train_len)
            for i in range(img_channels):
                train_loader[itr,jtr,i,:,:] = train_sequence[rand,:,:]

    for itr in range(test_loader_len):
        for jtr in range(batch_size):
            rand = randrange(test_len)
            for i in range(img_channels):
                test_loader[itr,jtr,i,:,:] = test_sequence[rand,:,:]
    
    plt.figure()
    imgplot = plt.imshow(train_loader[0,0,0])
    plt.show()

    plt.figure()
    imgplot = plt.imshow(test_loader[0,0,0])
    plt.show()

    with open(train_loader_file, 'wb') as f:
        np.save(f, train_loader)
    f.close()

    with open(test_loader_file, 'wb') as f:
        np.save(f, test_loader)
    f.close()


if __name__ == '__main__':
    main()