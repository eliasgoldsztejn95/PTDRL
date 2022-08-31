#!/usr/bin/env python3

import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import os

from IPython.display import clear_output
import matplotlib.pyplot as plt

import rospy
import task_env

#USE_CUDA = torch.cuda.is_available()

# Environment variables
num_obs = 16
num_actions = 12

dir_path = os.path.dirname(os.path.realpath(__file__))


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = list(zip(*samples))
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)

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
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state   = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action

class DDQN_Replay():
    def __init__(self, num_obs, num_actions, current_model, target_model, replay_buffer, env):

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.current_model = current_model
        self.target_model = current_model
        self.replay_buffer = replay_buffer
        self.env = env

        super(DDQN_Replay, self).__init__()
    
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())
    
    def action_to_params(self, action):
        if action == 0:
            params = { 'max_vel_x' : 0.15, 'min_vel_x' : -0.15, 'max_vel_trans': 0.15, 'path_distance_bias': 22}
        elif action == 1:
            params = { 'max_vel_x' : 0.3, 'min_vel_x' : -0.3, 'max_vel_trans': 0.3, 'path_distance_bias': 22}
        elif action == 2:
            params = { 'max_vel_x' : 1.0, 'min_vel_x' : -1.0, 'max_vel_trans': 1.0, 'path_distance_bias': 22}
        elif action == 3:
            params = { 'max_vel_x' : 1.5, 'min_vel_x' : -1.5, 'max_vel_trans': 1.0, 'path_distance_bias': 22}
        elif action == 4:
            params = { 'max_vel_x' : 0.15, 'min_vel_x' : -0.15, 'max_vel_trans': 0.15, 'path_distance_bias': 32}
        elif action == 5:
            params = { 'max_vel_x' : 0.3, 'min_vel_x' : -0.3, 'max_vel_trans': 0.3, 'path_distance_bias': 32}
        elif action == 6:
            params = { 'max_vel_x' : 1.0, 'min_vel_x' : -1.0, 'max_vel_trans': 1.0, 'path_distance_bias': 32}
        elif action == 7:
            params = { 'max_vel_x' : 1.5, 'min_vel_x' : -1.5, 'max_vel_trans': 1.0, 'path_distance_bias': 32}
        elif action == 8:
            params = { 'max_vel_x' : 0.15, 'min_vel_x' : -0.15, 'max_vel_trans': 0.15, 'path_distance_bias': 42}
        elif action == 9:
            params = { 'max_vel_x' : 0.3, 'min_vel_x' : -0.3, 'max_vel_trans': 0.3, 'path_distance_bias': 42}
        elif action == 10:
            params = { 'max_vel_x' : 1.0, 'min_vel_x' : -1.0, 'max_vel_trans': 1.0, 'path_distance_bias': 42}
        elif action == 11:
            params = { 'max_vel_x' : 1.5, 'min_vel_x' : -1.5, 'max_vel_trans': 1.0, 'path_distance_bias': 42}
        
        return params

    def test(self):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :return:
        """

        # Variables

        num_steps = 6000000

        max_episode = 101
        max_time_episode = 300
        #######################

        self.update_target()

        state = self.env.reset()

        episode = 1
        time_episode = 0

        ######################

        hits_per_episode = []
        time_per_episode = []

        hit = 0


        f_h = open(dir_path + "/hits.txt", "w")
        f_t = open(dir_path + "/time.txt", "w")


        for step in range(1, num_steps + 1):
            action = self.current_model.act(state, 0)
            if torch.is_tensor(action):
                action = action.item()
            action = 5
            print(f"Action is {action}")
            #next_state, reward, done = self.env.step(self.action_to_params([action]))
            next_state, reward, done = self.env.step([1, -1, 0, 0.5, 0.5, 0.5, 0.5])
            state = next_state

            time_episode += 1

            if reward == -30:
                hit += 1
            
            if time_episode >= max_time_episode:
                print("Max time!!!")
                done = 1
            
            if done:
                episode += 1
                state = self.env.reset()

                hits_per_episode.append(hit)
                a = np.asarray(hits_per_episode)
                with open(dir_path + "/hits.txt", "w") as f_h:
                    np.savetxt(f_h, a, fmt='%1.3f', newline=", ")

                time_per_episode.append(time_episode)
                b = np.asarray(time_per_episode)
                with open(dir_path + "/time.txt", "w") as f_t:
                    np.savetxt(f_t, b, fmt='%1.3f', newline=", ")

                time_episode = 0
                hit = 0
            
            if episode >= max_episode:
                break
            

        self.plot()


def main():

    current_model = DQN(num_obs, num_actions)
    target_model  = DQN(num_obs, num_actions)
    replay_buffer = NaivePrioritizedBuffer(10000)
    env = task_env.PtdrlTaskEnv()

    print("Loading prediction network")
    current_model.load_state_dict(torch.load(dir_path + '/checkpoint_dqn.pt'))
    current_model.eval()
    
    complete_model = DDQN_Replay(num_obs, num_actions, current_model, target_model, replay_buffer, env)
    
    complete_model.test()
 

if __name__ == '__main__':
    rospy.init_node('init_test')
    main()