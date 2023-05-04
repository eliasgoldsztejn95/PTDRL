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
from random import shuffle

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)


import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
#USE_CUDA = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

def create_trial_data():
    state_0 = np.zeros((10, 32))
    action_0 = np.zeros(32) + 1

    state_1 = np.zeros((10, 32))
    action_1 = np.zeros(32) + 0

    state_2 = np.zeros((10, 32))
    action_2 = np.zeros(32) + 0

    state_3 = np.zeros((10, 32))
    action_3 = np.zeros(32) + 1
    
    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "state_0.npy", state_0)
    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "action_0.npy", action_0)

    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "state_1.npy", state_1)
    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "action_1.npy", action_1)

    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "state_2.npy", state_2)
    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "action_2.npy", action_2)

    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "state_3.npy", state_3)
    np.save(dir_path + "/ptdrlhf" + "/state_action/" + "action_3.npy", action_3)


def load_batches(n_chunk, batch):

    x_chunk = np.load(dir_path + "/ptdrlhf" + "/state_action/" + "state_" + str(n_chunk) + ".npy")
    len_chunk = x_chunk.shape
    len_chunk = len_chunk[0]
    print(f"shape original: {x_chunk.shape}")
    x_chunk = x_chunk.reshape(int(len_chunk/batch), batch, 290)
    #print(f"shape after: {x_chunk.shape}")

    y_chunk = np.load(dir_path + "/ptdrlhf" + "/state_action/" + "action_" + str(n_chunk) + ".npy")
    #print(f"shape original: {y_chunk.shape}")
    y_chunk = y_chunk.reshape(int(len_chunk/batch), batch)
    #print(f"shape after: {y_chunk.shape}")

    return x_chunk, y_chunk


class DQN(nn.Module):
    def __init__(self, num_obs, num_actions):

        self.num_obs = num_obs
        self.num_actions = num_actions
        
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(self.num_obs , 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
    def forward(self, x):
        return self.layers(x)

    # def act(self, state):
    #     with torch.no_grad():
    #         state   = torch.cuda.FloatTensor(state).unsqueeze(0)
    #     q_value = self.forward(state)
    #     action  = q_value.max(1)[1].data[0]
    #     return action



def train(net, batch, len_data, epochs):

    # Set optimization parameters
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
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
            x_batches, y_batches = load_batches(n_chunk[0], batch) # Load chunk of data in batches
            #print(f"x_batches: {x_batches}")
            #print(f"y_batches: {y_batches}")
            len_batches = x_batches.shape
            len_batches = len_batches[0]
            #print(f"x_batches.shape: {x_batches.shape}")
            

            # Iterate over batches
            for n_batch in range(len_batches):
                #x = torch.cuda.FloatTensor(x_batches[n_batch])
                x = torch.FloatTensor(x_batches[n_batch]).to(device)
                #x = x.type(torch.cuda.LongTensor)
                y = torch.LongTensor(y_batches[n_batch]).to(device)
                #y = y.type(torch.cuda.LongTensor)

                #print(f"x: {x}")
                #print(f"y: {y}")
                out = net(x)   
                #print(f"out: {out}")              # input x and predict based on x
                loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients

                # show learning process
                prediction = torch.max(out, 1)[1]
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
        x_batches, y_batches = load_batches(n_chunk[0], batch) # Load chunk of data in batches
        len_batches = x_batches.shape
        len_batches = len_batches[0]

        # Iterate over batches
        for n_batch in range(len_batches):
            x = torch.FloatTensor(x_batches[n_batch]).to(device)
            y = torch.LongTensor(y_batches[n_batch]).to(device)

            with torch.no_grad():
                out = net(x)                 # input x and predict based on x

            # show learning process
            prediction = torch.max(out, 1)[1]
            pred_y = prediction.cpu().data.numpy()
            target_y = y.cpu().data.numpy()
            accuracy += float((pred_y == target_y).astype(int).sum()) / float(target_y.size)

        accuracy /= len_batches
    accuracy /= len(n_test)
    print(f"Test accuracy = {accuracy}")

    if accuracy > best_acc:
        torch.save(net.state_dict(), dir_path + "/ptdrlhf/checkpoints/state_action" + '/checkpoint_supervised_dqn' + str(round(accuracy*100)) + '.pth')
        return accuracy
    return best_acc

def main():

    #create_trial_data()
    net = DQN(num_obs=290, num_actions=4)
    if USE_CUDA:
        net = DQN(num_obs=290, num_actions=4).cuda()
    #     print(net)
    #     for name, param in net.named_parameters():
    #         if name == "layers.0.weight":
    #             print(f"First layer: {param}")
    #         if name == "layers.0.bias":
    #             print(f"First bias: {param}")
    #         if name == "layers.4.weight":
    #             print(f"Last layer: {param}")
    # state_dict = torch.load(dir_path + "/ptdrlhf/checkpoints/state_action" + '/checkpoint_supervised_dqn' + "95" + '.pth')
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
    #             print(f"Last layer: {param}")
    #net.load_state_dict(torch.load(dir_path + "/ptdrlhf/checkpoints/state_action" + '/checkpoint_supervised_dqn' + "95" + '.pth'))
    #print(net)

    net = train(net = net, batch = 8, len_data=2, epochs = 5)

if __name__ == '__main__':
    main()
