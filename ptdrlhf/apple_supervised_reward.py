#!/usr/bin/env python3

# https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb

import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import os
import random
import copy
from random import shuffle

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)


import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

def create_trial_data():
    state_0 = np.zeros((32, 290))
    action_0 = np.zeros(32) + 1
    reward_0 = np.zeros(32) + 0

    state_1 = np.zeros((32, 290))
    action_1 = np.zeros(32) + 0
    reward_1 = np.zeros(32) + 1

    state_2 = np.zeros((32, 290))
    action_2 = np.zeros(32) + 0
    reward_2 = np.zeros(32) + 0

    state_3 = np.zeros((32, 290))
    action_3 = np.zeros(32) + 1
    reward_3 = np.zeros(32) + 1
    
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "state_0.npy", state_0)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "action_0.npy", action_0)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "reward_0.npy", reward_0)

    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "state_1.npy", state_1)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "action_1.npy", action_1)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "reward_1.npy", reward_1)

    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "state_2.npy", state_2)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "action_2.npy", action_2)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "reward_2.npy", reward_2)

    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "state_3.npy", state_3)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "action_3.npy", action_3)
    np.save(dir_path + "/ptdrlhf" + "/apple_state_action_reward/" + "reward_3.npy", reward_3)


def load_batches(n_chunk, batch):

    state_chunk = np.load(dir_path + "/ptdrlhf" + "/state_action_reward/" + "state_" + str(n_chunk) + ".npy")
    len_chunk = state_chunk.shape
    len_chunk = len_chunk[0]
    #print(f"state_chunk {state_chunk}")
    state_chunk = state_chunk.reshape(int(len_chunk/batch), batch, 290)
    #print(f"shape after: {x_chunk.shape}")

    action_chunk = np.load(dir_path + "/ptdrlhf" + "/state_action_reward/" + "action_" + str(n_chunk) + ".npy")
    #print(f"action_chunk: {action_chunk}")    
    #print(f"shape after: {y_chunk.shape}")
    #print(f"action_chunk.max()  {action_chunk.max() }")
    #print(f"action_hot_encoded_chunk {action_hot_encoded_chunk}")
    action_chunk = action_chunk.reshape(int(len_chunk/batch), batch)
    #print(f"action_hot_encoded_chunk.shape {action_hot_encoded_chunk.shape}")

    reward_chunk = np.load(dir_path + "/ptdrlhf" + "/state_action_reward/" + "reward_" + str(n_chunk) + ".npy")
    #print(f"shape original: {y_chunk.shape}")
    reward_chunk = reward_chunk.reshape(int(len_chunk/batch), batch)
    #print(f"shape after: {y_chunk.shape}")

    return state_chunk, action_chunk, reward_chunk

class REWARD(nn.Module):
    def __init__(self, num_obs, num_actions):

        self.num_obs = num_obs
        self.num_actions = num_actions
        
        super(REWARD, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(self.num_obs + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.ReLU(),
            nn.Linear(4, 1)
        )


    def forward(self, x0, x1):
        x1 = torch.unsqueeze(x1, 1)
        x = torch.cat((x0, x1), dim=1)
        return self.layers(x)



def train(net, batch, len_data, epochs):

    # Set optimization parameters
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.BCEWithLogitsLoss()  # the target label is NOT an one-hotted
    best_acc = 0

    # Iterate epochs
    for epoch in range(epochs):
        accuracy = 0
        # Randomize data
        n_data = [[i] for i in range(len_data)]
        shuffle(n_data)
        n_train = n_data[0:int(len_data*0.8)]
        #print(f"n_train: {n_train}")
        n_test = n_data[int(len_data*0.8)::]
        #print(f"n_test: {n_test}")

        # Iterate over chunks in train set
        for n_chunk in n_train:
            #print(f"n_chunk: {n_chunk[0]}")
            x0_batches, x1_batches, y_batches = load_batches(n_chunk[0], batch) # Load chunk of data in batches
            #print(f"x_batches: {x_batches}")
            #print(f"y_batches: {y_batches}")
            len_batches = x0_batches.shape
            len_batches = len_batches[0]
            #print(f"x_batches.shape: {x_batches.shape}")
            

            # Iterate over batches
            for n_batch in range(len_batches):
                #x = torch.cuda.FloatTensor(x_batches[n_batch])
                x0 = torch.FloatTensor(x0_batches[n_batch]).to(device)
                #x = x.type(torch.cuda.LongTensor)
                x1 = torch.FloatTensor(x1_batches[n_batch]).to(device)
                #y = y.type(torch.cuda.LongTensor)
                y = torch.FloatTensor(y_batches[n_batch]).to(device)

                #print(f"x: {x}")
                #print(f"y: {y}")
                out = net(x0, x1)  
                out = torch.squeeze(out)
                #print(f"out: {torch.squeeze(out)}")
                #print(f"out: {out}")              # input x and predict based on x
                #print(f"y: {y}")
                loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

                # show learning process
                print(f"out: {out}")
                prediction = (out >= 0.5).float()
                #prediction = out.max(1)[1].data
                pred_y = prediction.cpu().data.numpy()
                print(f"prediction {pred_y}")
                target_y = y.cpu().data.numpy()
                print(f"target_y: {target_y}")
                accuracy += float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            accuracy /= len_batches

        print(f"Train accuracy epoch = {accuracy/len(n_train)}")
        
        best_acc = test(net, best_acc, n_test, batch)

    return net

def test(net, best_acc, n_test, batch):

    # Iterate over chunks in test set
    accuracy = 0
    for n_chunk in n_test:
        x0_batches, x1_batches, y_batches = load_batches(n_chunk[0], batch) # Load chunk of data in batches
        len_batches = x0_batches.shape
        len_batches = len_batches[0]

        # Iterate over batches
        for n_batch in range(len_batches):
            #x = torch.cuda.FloatTensor(x_batches[n_batch])
            x0 = torch.FloatTensor(x0_batches[n_batch]).to(device)
            #x = x.type(torch.cuda.LongTensor)
            x1 = torch.FloatTensor(x1_batches[n_batch]).to(device)
            #y = y.type(torch.cuda.LongTensor)
            y = torch.FloatTensor(y_batches[n_batch]).to(device)

            with torch.no_grad():
                out = net(x0, x1)                  # input x and predict based on x
            out = torch.squeeze(out)

            # show learning process
            prediction = (out >= 0.5).float()
            pred_y = prediction.cpu().data.numpy()
            target_y = y.cpu().data.numpy()
            accuracy += float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

        accuracy /= len_batches
    accuracy /= len(n_test)
    print(f"Test accuracy = {accuracy}")

    if accuracy > best_acc:
        torch.save(net.state_dict(), dir_path + "/ptdrlhf/checkpoints/apple_state_action_reward" + '/checkpoint_supervised_reward' + str(round(accuracy*100)) + '.pth')
        return accuracy
    return best_acc

def main():

    create_trial_data()
    net = REWARD(num_obs=290, num_actions=4)
    if USE_CUDA:
        net = REWARD(num_obs=290, num_actions=4).cuda()

    net = train(net = net, batch = 8, len_data=4, epochs = 10)

if __name__ == '__main__':
    main()
























    # net = DQN(num_obs=290, num_actions=4)
    # if USE_CUDA:
    #     net = DQN(num_obs=290, num_actions=4).cuda()
    #     print(net)
    #     for name, param in net.named_parameters():
    #         if name == "layers.0.weight":
    #             print(f"First layer: {param}")
    #         if name == "layers.0.bias":
    #             print(f"First bias: {param}")
    #         if name == "layers.4.weight":
    #             print(f"Last layer: {param}")
    # state_dict = torch.load(dir_path + "/ptdrlhf/checkpoints/state_action" + '/checkpoint_supervised_dqn' + "94" + '.pth')
    # print(f"state_dict fisrt layer: {state_dict['layers.0.weight']}")
    # print(f"state_dict fisrt bias: {state_dict['layers.0.bias']}")
    # with torch.no_grad():
    #     for name, param in net.named_parameters():
    #         if name == "layers.0.weight":
    #             print("herhe")
    #             param.copy_(state_dict['layers.0.weight'])
    #         if name == "layers.0.bias":
    #             print("herhe")
    #             param.copy_(state_dict['layers.0.bias'])
    #     for name, param in net.named_parameters():
    #         if name == "layers.0.weight":
    #             print(f"First layer: {param}")
    #         if name == "layers.0.bias":
    #             print(f"First bias: {param}")
    #         if name == "layers.4.weight":
