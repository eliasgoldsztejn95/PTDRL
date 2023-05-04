"""
Supervised state-action network. Output is probabilty distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from scipy.special import softmax

class SFT(nn.Module):
    def __init__(self, num_obs, num_actions):

        self.num_obs = num_obs
        self.num_actions = num_actions
        
        super(SFT, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(self.num_obs , 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
    def forward(self, x):
        return self.layers(x)

    def act(self, state):
        with torch.no_grad():
            state   = torch.cuda.FloatTensor(state).unsqueeze(0)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        soft = nn.Softmax(dim=1)
        return soft(q_value)