from cgi import test
from math import cos
from turtle import title
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse
import os
from models.vae import VAE
from os.path import join, exists
from skimage import color
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.cm as cm
from random import randrange
import matplotlib.animation as animation

import phydnet_predict

# Path
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_image = dir_path + "/loader_image"

reload_file_input = dir_path + "/exp_dir/vae_input/best.tar"
reload_file_prediction = dir_path + "/exp_dir/vae_prediction/best.tar"

train_loader_file = dir_path_image + "/costmap1"

dir_path_checkpoint = dir_path + '/best_networks/encoder_phydnet_128_11.pth'

cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

img_channels = 3

def main():

    with open(train_loader_file + ".npy", 'rb') as f:
        costmap = np.load(f)
    
    for i in range(740,1000,5):

        #####################################################
        # Costmap, image, video #############################
        start = i
        image = np.zeros([1,3,64,64])
        video = np.zeros([1,10,1,60,60])
        image_p = np.zeros([1,3,64,64])

        for i in range(3):
            image[0,i,0:60,0:60] = costmap[start,:,:]/100
        
        video[0,:,0,:,:] = costmap[start - 10:start,:,:]/100
        #####################################################
        #####################################################


        #####################################################
        # Init NN's #########################################
        phydnet =  phydnet_predict.PhydNet(dir_path_checkpoint)

        vae_input = VAE(3, 4).to(device)
        vae_prediction = VAE(3, 4).to(device)

        state_input = torch.load(reload_file_input)
        vae_input.load_state_dict(state_input['state_dict'])

        state_prediction = torch.load(reload_file_prediction)
        vae_prediction.load_state_dict(state_prediction['state_dict'])
        #####################################################
        #####################################################


        #####################################################
        # Predict and encode ################################
        
        # Predict video #
        predictions = phydnet.predict(video)
        for i in range(img_channels):
            image_p[0,i,0:60,0:60] = np.squeeze(predictions[:,-1,:,:])

        # Encode input image
        image = torch.tensor(image)
        image = image.to(device, dtype=torch.float)
        image_o, mu, logvar = vae_input(image)

        # Encode predicted image
        image_p = torch.tensor(image_p)
        image_p = image_p.to(device, dtype=torch.float)
        image_p_o, mu, logvar  = vae_prediction(image_p)
        #####################################################
        #####################################################

        #####################################################
        # Show videos #######################################
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Videos')

        print("Input video")
        frames1 = [] # for storing the generated images
        for i in range(10):
            frames1.append([ax1.imshow(np.squeeze(video[0,i,0,:,:]), cmap=cm.Greys_r,animated=True)])
        ax1.set_title("Input")

        print("Actual future video")
        frames2 = [] # for storing the generated images
        for i in range(10):
            frames2.append([ax2.imshow(np.squeeze(costmap[start + i,:,:]/100), cmap=cm.Greys_r,animated=True)])
        ax2.set_title("Future")

        print("Prediction video")
        frames3 = [] # for storing the generated images
        for i in range(10):
            frames3.append([ax3.imshow(np.squeeze(predictions[0,i,:,:]), cmap=cm.Greys_r,animated=True)])
        ax3.set_title("Predicted")

        ani = animation.ArtistAnimation(fig, frames1, interval=100, blit=True)
        ani2 = animation.ArtistAnimation(fig, frames2, interval=100, blit=True)
        ani3 = animation.ArtistAnimation(fig, frames3, interval=100, blit=True)

        #####################################################
        #####################################################

        #####################################################
        # Show images #######################################

        fig2, (ax11, ax22, ax33) = plt.subplots(1, 3)
        fig2.suptitle('Images')

        im1 = Image.fromarray(np.squeeze(image_o[0,0].cpu().data.numpy())*256)
        imgplot = ax11.imshow(im1)
        ax11.set_title("Decoded input")

        im2 = Image.fromarray(np.squeeze((costmap[start + 10,:,:]/100)*256))
        imgplot = ax22.imshow(im2)
        ax22.set_title("Future")

        im3 = Image.fromarray(np.squeeze(image_p_o[0,0].cpu().data.numpy())*256)
        imgplot = ax33.imshow(im3)
        ax33.set_title("Decoded prediction")

        #####################################################
        #####################################################

        plt.show()
        

if __name__ == '__main__':
    main()