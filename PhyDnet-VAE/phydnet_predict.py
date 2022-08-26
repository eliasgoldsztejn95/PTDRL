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

dir_path = os.path.dirname(os.path.realpath(__file__))

class PhydNet():
    def __init__(self, checkpoint):

        self.target_length = 10 # predict the next 10 frames

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.checkpoint = checkpoint
        self.phycell  =  PhyCell(input_shape=(15,15), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=self.device) #F_hidden_dims 49
        self.convcell =  ConvLSTM(input_shape=(15,15), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=self.device)   #hidden_dims 128,128,64
        self.encoder  = EncoderRNN(self.phycell, self.convcell, self.device)

        print("Loading prediction network")
        self.encoder.load_state_dict(torch.load(self.checkpoint ))
        self.encoder.eval()
    
    def predict(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs)
            input_tensor = obs.to(self.device, dtype=torch.float)
            input_length = input_tensor.size()[1]

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = self.encoder(input_tensor[:,ei,:,:,:], (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(self.target_length):
                decoder_output, decoder_hidden, output_image,_,_ = self.encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)

        return predictions

