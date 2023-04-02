""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import os
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
from random import randrange

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
args = parser.parse_args()


    
# Load loaders
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/costmap_odom_video"
print(dir_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# constants
BSIZE = 16
SEQ_LEN = 32
IMG_CH = 3
BF_TRAIN = 150
BF_TEST = 30
epochs = 800

# Loading VAE
vae_file = join(args.logdir, 'vae_inflation', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Loading model
rnn_dir = join(args.logdir, 'mdrnn_ptdrl')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


# Data Loading
data_counter = 1
num_data = 2
data_len = 0
train_len = 0
file_costmap = "costmap_sync"
file_vel = "vel_sync"
file_inf = "inf_sync"


def load_data():
    global data_counter
    global data_len
    global train_len
    # Open data
    with open(dir_path + "/" + file_costmap + "_" + str(data_counter) + ".npy", 'rb') as f:
        data_costmap = np.load(f)
    with open(dir_path + "/" + file_vel + "_" + str(data_counter) + ".npy", 'rb') as f:
        data_vel = np.load(f)
    with open(dir_path + "/" + file_inf + "_" + str(data_counter) + ".npy", 'rb') as f:
        data_inf = np.load(f)

    train_len = int(len(data_vel)*0.8) -BF_TRAIN - SEQ_LEN - 1
    data_len = len(data_vel) - SEQ_LEN - BF_TEST - 1
    data_counter += 1
    data_counter = data_counter % num_data
    print(f"Loading data of size: {data_len}")
    return data_costmap, data_vel, data_inf

    
data_costmap, data_vel, data_inf =  load_data()


buffer_train_index = 0
buffer_train_size = BF_TRAIN
buffer_test_index = train_len
buffer_test_size = BF_TEST

def update_buffers_and_data():
    global buffer_train_index
    global buffer_train_size
    global buffer_test_index
    global buffer_test_size

    # Increase counter 
    buffer_train_index += buffer_train_size
    buffer_test_index += buffer_test_size

    # If finished train data load next data and reset
    if buffer_train_index > train_len:
        buffer_train_index = 0
        buffer_test_index = 0
        load_data()

    # In case we dont have a full buffer_train_size chunk
    buffer_train_size_actual = buffer_train_size if (train_len - buffer_train_index) > buffer_train_size else (train_len - buffer_train_index)

    # In case we passed the training data length
    buffer_test_index = buffer_test_index % data_len
    if buffer_test_index < train_len:
        buffer_test_index = train_len
    
    # In case we dont have a full buffer_test_size chunk
    buffer_test_size_actual = buffer_test_size if (data_len - buffer_test_index) > buffer_test_size else (data_len - buffer_test_index)

    return buffer_train_size_actual,buffer_test_size_actual

def load_next_buffer(buffer_train_index, buffer_train_size_actual, buffer_test_index, buffer_test_size_actual):

    train_sequence = np.zeros([buffer_train_size,SEQ_LEN + 1, 64, 64])
    test_sequence = np.zeros([buffer_test_size,SEQ_LEN + 1, 64, 64])
    train_action = np.zeros([buffer_train_size,SEQ_LEN, 3])
    test_action = np.zeros([buffer_test_size,SEQ_LEN, 3])

    train_loader_obs = np.zeros([buffer_train_size, BSIZE, SEQ_LEN , IMG_CH, 64, 64])
    train_loader_next_obs = np.zeros([buffer_train_size, BSIZE, SEQ_LEN , IMG_CH, 64, 64])
    train_loader_action = np.zeros([buffer_train_size, BSIZE, SEQ_LEN , 3])

    test_loader_obs = np.zeros([buffer_test_size, BSIZE, SEQ_LEN , IMG_CH, 64, 64])
    test_loader_next_obs = np.zeros([buffer_test_size, BSIZE, SEQ_LEN , IMG_CH, 64, 64])
    test_loader_action = np.zeros([buffer_test_size, BSIZE, SEQ_LEN , 3])


    # Store in train and test
    for itr in range(buffer_train_index, buffer_train_index + buffer_train_size):
        train_sequence[itr - buffer_train_index,:,0:60,0:60] = data_costmap[itr: itr + SEQ_LEN + 1,:,:]/100
        train_action[itr - buffer_train_index,:,0:2] = data_vel[itr: itr + SEQ_LEN,:]
        train_action[itr - buffer_train_index,:,2] = data_inf[itr: itr + SEQ_LEN]
    for itr in range(buffer_test_index, buffer_test_index + buffer_test_size):
        #print(f" itr, itr + SEQ_LEN + 1  : {itr}, {itr + SEQ_LEN + 1}")
        #print(f" data_costmap {data_costmap[itr: itr + SEQ_LEN + 1,:,:].shape}")
        test_sequence[itr - buffer_test_index,:,0:60,0:60] = data_costmap[itr: itr + SEQ_LEN + 1,:,:]/100
        test_action[itr - buffer_test_index,:,0:2] = data_vel[itr: itr + SEQ_LEN,:]
        test_action[itr - buffer_test_index,:,2] = data_inf[itr: itr + SEQ_LEN]

    # Store in loaders
    for itr_len in range(buffer_train_size):
        for itr_batch in range(BSIZE):
            rand = randrange(buffer_train_size)
            for itr_channel in range(IMG_CH):
                train_loader_obs[itr_len,itr_batch,:,itr_channel,:,:] = train_sequence[rand,0:-1,:,:]
                train_loader_next_obs[itr_len,itr_batch,:,itr_channel,:,:] = train_sequence[rand,1::,:,:]
            train_loader_action[itr_len,itr_batch,:,:] = train_action[rand,:,:]

    for itr_len in range(buffer_test_size):
        for itr_batch in range(BSIZE):
            rand = randrange(buffer_test_size)
            for itr_channel in range(IMG_CH):
                test_loader_obs[itr_len,itr_batch,:,itr_channel,:,:] = test_sequence[rand,0:-1,:,:]
                test_loader_next_obs[itr_len,itr_batch,:,itr_channel,:,:] = test_sequence[rand,1::,:,:]
            test_loader_action[itr_len,itr_batch,:,:] = test_action[rand,:,:]
    
    return train_loader_obs, train_loader_next_obs, train_loader_action, test_loader_obs, test_loader_next_obs, test_loader_action

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs

def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)

    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        print("reward included")
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epoch, train, include_reward): # pylint: disable=too-many-locals

    buffer_train_size_actual, buffer_test_size_actual = update_buffers_and_data()

    train_loader_obs, train_loader_next_obs, train_loader_action, test_loader_obs, test_loader_next_obs, test_loader_action = load_next_buffer(buffer_train_index, buffer_train_size_actual, buffer_test_index, buffer_test_size_actual)

    """ One pass through the data """
    if train:
        mdrnn.train()
        loader_obs = train_loader_obs
        loader_next_obs = train_loader_next_obs
        loader_action = train_loader_action
        print(f"Buffer_index is {buffer_train_index} from maximum of: {train_len}")
    else:
        mdrnn.eval()
        loader_obs = test_loader_obs
        loader_next_obs = test_loader_next_obs
        loader_action = test_loader_action
        print(f"Buffer_index is {buffer_test_index} from maximum of: {data_len - train_len}")
    
    len_loader = len(loader_obs)


    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len_loader, desc="Epoch {}".format(epoch))
    for i in range(len_loader):
        obs = torch.cuda.FloatTensor(loader_obs[i])
        next_obs = torch.cuda.FloatTensor(loader_next_obs[i])
        action = torch.cuda.FloatTensor(loader_action[i])
        reward = torch.cuda.FloatTensor(np.zeros([16,32]))
        terminal = torch.cuda.FloatTensor(np.zeros([16,32]))

        # print(f"obs shape: {obs.shape}")
        # print(obs)
        # print(f"action shape: {action.shape}")
        # print(action)
        # print(f"reward shape: {reward.shape}")
        # print(reward)
        # print(f"terminal shape: {terminal.shape}")
        # print(terminal)

        ########## Plot # Batch=16, Sequence=32, Channels=3, 64, 64
        ## next_obs is one frame movement of obs
        # print(f"is obs + 1 == to next_obs?: {torch.all(obs[0,6,1,:,:].eq(next_obs[0,5,1,:,:]))}")
        ##print(obs[0,0,0,:,:].cpu().data.numpy())
        # fig2, (ax11, ax22) = plt.subplots(1, 2)
        # fig2.suptitle('Images')

        # im1 = Image.fromarray(np.fliplr(np.squeeze(obs[10,0,1,:,:].cpu().data.numpy()))*255)
        # print(f"Actions is {action[10,1]}")
        # imgplot = ax11.imshow(im1)
        # ax11.set_title("obs1")


        # im3 = Image.fromarray(np.fliplr(np.squeeze(obs[10,0,1,:,:].cpu().data.numpy()))*255)
        # imgplot = ax22.imshow(im3)
        # ax22.set_title("obs2")

        # plt.show(block=False)
        # plt.pause(10)
        # plt.close()
        # plt.show()

        #############################3

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len_loader


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)


cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break
