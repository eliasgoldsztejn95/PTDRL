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

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/loader_video"
print(dir_path)

train_loader_file = dir_path
train_loader_file += "/train_loader.npy"

test_loader_file = dir_path
test_loader_file += "/test_loader.npy"

epoch_count = 0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default= dir_path + '/data/')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--nepochs', type=int, default=401, help='nb of epochs') #250 is enough. 2001
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=10, help='')
parser.add_argument('--save_name', type=str, default='phydnet', help='')
args = parser.parse_args()


with open(train_loader_file, 'rb') as f:
    train_loader = np.load(f)
f.close()

with open(test_loader_file, 'rb') as f:
    test_loader = np.load(f)
f.close()


constraints = torch.zeros((49,7,7)).to(device) #49 7 7
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1    

def get_batch(loader, itr):
    return [torch.tensor(loader[itr,:,0:10,:,:,:]), torch.tensor(loader[itr,:,10::,:,:,:])]


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,teacher_forcing_ratio):                
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    for ei in range(input_length-1): 
        encoder_output, encoder_hidden, output_image,_,_ = encoder(input_tensor[:,ei,:,:,:], (ei==0) )
        loss += criterion(output_image,input_tensor[:,ei+1,:,:,:])

    decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
        target = target_tensor[:,di,:,:,:]
        loss += criterion(output_image,target)
        if use_teacher_forcing:
            decoder_input = target # Teacher forcing    
        else:
            decoder_input = output_image

    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0,encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion(m, constraints) # constrains is a precomputed matrix   
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, nepochs, print_every=10,eval_every=10,name=''):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2,factor=0.1,verbose=True)
    criterion = nn.MSELoss()
    print(len(train_loader))
    print(len(test_loader))
    
    for epoch in range(0, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
        for itr in range(len(train_loader)):
            out = get_batch(train_loader, itr)
            input_tensor = out[0].to(device, dtype=torch.float)

            target_tensor = out[1].to(device, dtype=torch.float)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio)                                   
            loss_epoch += loss
                      
        train_losses.append(loss_epoch)        
        if (epoch+1) % print_every == 0:
            print('epoch ',epoch,  ' loss ',loss_epoch, ' time epoch ',time.time()-t0)
            
        if (epoch+1) % eval_every == 0:
            mse, mae,ssim = evaluate(encoder,test_loader) 
            scheduler_enc.step(mse)             
            torch.save(encoder.state_dict(), dir_path + '/save2/encoder_{}'.format(name) + '_' + str(int(mse)) + '_' + str(int(loss_epoch)) + '.pth')                           
    return train_losses

    
def evaluate(encoder,loader):
    global epoch_count
    epoch_count += 1
    print(epoch_count)
    total_mse, total_mae,total_ssim,total_bce = 0,0,0,0
    t0 = time.time()

    with torch.no_grad():
        for i in range(60,70,1):
            out = get_batch(loader,i)
            input_tensor = out[0].to(device, dtype=torch.float)
            target_tensor = out[1].to(device, dtype=torch.float)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)
            
            if (epoch_count % 1) == 0:
                frames = [] # for storing the generated images
                fig = plt.figure()
                rand = randrange(16)
                for i in range(10):
                    frames.append([plt.imshow(np.squeeze(predictions[rand,i,0,:,:]*0), cmap=cm.Greys_r,animated=True)])
                for i in range(10):
                    frames.append([plt.imshow(np.squeeze(input[rand,i,0,:,:]), cmap=cm.Greys_r,animated=True)])
                for i in range(10):
                    frames.append([plt.imshow(np.squeeze(predictions[rand,i,0,:,:]*0), cmap=cm.Greys_r,animated=True)])
                for i in range(10):
                    frames.append([plt.imshow(np.squeeze(target[rand,i,0,:,:]), cmap=cm.Greys_r,animated=True)])
                for i in range(10):
                    frames.append([plt.imshow(np.squeeze(predictions[rand,i,0,:,:]*0), cmap=cm.Greys_r,animated=True)])
                for i in range(10):
                    frames.append([plt.imshow(np.squeeze(predictions[rand, i,0,:,:]), cmap=cm.Greys_r,animated=True)])
            
                ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
                plt.show()

            mse_batch = np.mean((predictions-target)**2 , axis=(0,1,2)).sum()
            mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1,2)).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            
            for a in range(0,target.shape[0]):
                for b in range(0,target.shape[1]):
                    total_ssim += ssim(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0]*target.shape[1]) 

            
            cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size*target_length)
            total_bce +=  cross_entropy

     
    print('eval mse ', total_mse/len(loader),  ' eval mae ', total_mae/len(loader),' eval ssim ',total_ssim/len(loader), ' time= ', time.time()-t0)        
    return total_mse/len(loader),  total_mae/len(loader), total_ssim/len(loader)

#dir_path_checkpoint = dir_path + '/save/encoder_phydnet.pth'

#prediction_module =  phydnet_predict.PhydNet(dir_path_checkpoint)

phycell  =  PhyCell(input_shape=(15,15), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) #F_hidden_dims 49
convcell =  ConvLSTM(input_shape=(15,15), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   #hidden_dims 128,128,64
encoder  = EncoderRNN(phycell, convcell, device)
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   
print('phycell ' , count_parameters(phycell) )
print('convcell ' , count_parameters(convcell) ) 
print('encoder ' , count_parameters(encoder) ) 

#trainIters(encoder,args.nepochs,print_every=args.print_every,eval_every=args.eval_every,name=args.save_name)

encoder.load_state_dict(torch.load(dir_path + '/save2/encoder_phydnet_128_11.pth'))
encoder.eval()
mse, mae,ssim = evaluate(encoder,test_loader) 
