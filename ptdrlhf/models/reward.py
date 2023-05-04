"""
Reward network
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
import numpy as np

class REWARD(nn.Module):
    def __init__(self, num_obs, num_actions):

        self.num_obs = num_obs
        self.num_actions = num_actions
        
        super(REWARD, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(self.num_obs , 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

        self.layer_norm = nn.LayerNorm(self.num_actions)

        self.state_action_layers = nn.Sequential(
            nn.Linear((self.num_actions)*2 , 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x0, x1):
        x0_out = self.layers(x0)
        x0_out_normalized = self.layer_norm(x0_out)
        x = torch.cat((x0_out_normalized, x1), dim=1)
        return self.state_action_layers(x)

    def act(self, state, action):
        action_hot_encoded = np.zeros(4)
        action_hot_encoded[action] = 1

        with torch.no_grad():
            state = torch.cuda.FloatTensor(state).unsqueeze(0)
            action_hot_encoded = torch.cuda.FloatTensor(action_hot_encoded).unsqueeze(0)
        reward = self.forward(state, action_hot_encoded)
        reward = torch.squeeze(reward)
        #print(f"reward: {reward}")
        #print(f"reward thr: {(reward >= 0.5).float()}")
        return (reward >= 0.5).float()
