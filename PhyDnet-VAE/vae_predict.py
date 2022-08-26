""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
import os
import numpy as np

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--logdir', default='exp_dir',type=str, help='Directory where results are logged')


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/loader_image"
print(dir_path)

train_loader_file = dir_path
train_loader_file += "/train_loader.npy"

test_loader_file = dir_path
test_loader_file += "/test_loader.npy"

args = parser.parse_args()
cuda = torch.cuda.is_available()

with open(train_loader_file, 'rb') as f:
    train_loader = np.load(f)
f.close()

with open(test_loader_file, 'rb') as f:
    test_loader = np.load(f)
f.close()

torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

vae_dir = join(args.logdir, 'vae_input')

reload_file = join(vae_dir, 'best.tar')


def get_batch(loader, itr):
    return torch.tensor(loader[itr,:,:,:])

def main():

    model = VAE(3, LSIZE).to(device)

    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
            ", with test error {}".format(
                state['epoch'],
                state['precision']))
    model.load_state_dict(state['state_dict'])

    batch_idx = 863
    data = get_batch(train_loader, batch_idx)

    # Show actual image #
    plt.figure()
    imgplot = plt.imshow(data[0,2])
    plt.show()
    #####################
    print(data.shape)
    data = data.to(device, dtype=torch.float)
    recon_batch, mu, logvar = model(data)

    # Show reconstructed image #
    plt.figure()
    imgplot = plt.imshow(recon_batch[0,2].cpu().data.numpy())
    plt.show()
    ############################

if __name__ == '__main__':
    main()