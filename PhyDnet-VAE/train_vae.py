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
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
import os
import numpy as np

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', default='exp_dir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/loader_image"
print(dir_path)

train_loader_file = dir_path
train_loader_file += "/train_loader_1.npy"

test_loader_file = dir_path
test_loader_file += "/test_loader_1.npy"

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


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

# dataset_train = RolloutObservationDataset('datasets/carracing',
#                                           transform_train, train=True)
# dataset_test = RolloutObservationDataset('datasets/carracing',
#                                          transform_test, train=False)
# train_loader = torch.utils.data.DataLoader(
#     dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
# test_loader = torch.utils.data.DataLoader(
#     dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)


model = VAE(3, LSIZE).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

def get_batch(loader, itr):
    return torch.tensor(loader[itr,:,:,:])

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    #print(recon_x.shape)
    #print(x.shape)
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):
    print("here")
    """ One training epoch """
    model.train()
    #dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx in range(len(train_loader)):
        data = get_batch(train_loader, batch_idx)
        data = data.to(device, dtype=torch.float)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)*32,
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)*32))


def test():
    """ One test epoch """
    model.eval()
    #dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for batch_idx in range(len(test_loader)):
            data = get_batch(train_loader, batch_idx)
            #print(data[0,2])
            plt.figure()
            imgplot = plt.imshow(data[0,2])
            plt.show()
            data = data.to(device, dtype=torch.float)
            recon_batch, mu, logvar = model(data)
            #print(recon_batch[0,2])
            #print(recon_batch.shape)



            plt.figure()
            imgplot = plt.imshow(recon_batch[0,2].cpu().data.numpy())
            plt.show()

            #print(mu.shape)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader)*32
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

for epoch in range(1, args.epochs + 1):
    #train(epoch)
    test_loss = test()
    # scheduler.step(test_loss)
    # earlystopping.step(test_loss)

    # # checkpointing
    # best_filename = join(vae_dir, 'best.tar')
    # filename = join(vae_dir, 'checkpoint.tar')
    # is_best = not cur_best or test_loss < cur_best
    # if is_best:
    #     cur_best = test_loss

    # save_checkpoint({
    #     'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #     'precision': test_loss,
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'earlystopping': earlystopping.state_dict()
    # }, is_best, filename, best_filename)



    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
