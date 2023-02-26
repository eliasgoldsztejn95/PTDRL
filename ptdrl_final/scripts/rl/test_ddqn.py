#!/usr/bin/env python3

import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import os

from numpy import inf
from scipy import stats
import csv

from IPython.display import clear_output
import matplotlib.pyplot as plt

import rospy
import task_env
from pathlib import Path

USE_CUDA = torch.cuda.is_available()

# Environment variables
num_obs = 290
num_actions = 4

dir_path = os.path.dirname(os.path.realpath(__file__))

global test_file
test_file = dir_path + "/score_ddqn_mdrnn_a_1"

environments = {0:"open", 1:"door", 2:"curve", 3:"obstacles"}
dir_path_context = dir_path + "/networks/context_door.ckpt"

###### Context predictor ##########
class Predictor():
    def __init__(self):
        self.history =[]
        self.model = nn.Sequential(
		nn.Linear(720, 128),
		nn.ReLU(),
		nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 4),)
        self.model.load_state_dict(torch.load(dir_path_context))
        self.context_type = 0

        self.y = np.zeros([4])
        

    def update_context(self, scan):
            # print np.mean(msg.ranges)
            # change context type here
            #print(scan.ranges)
            scann = list(scan.ranges)
            scan_not_inf = []
            for i in scann:
                if i < 3.5:
                    scan_not_inf.append(i)
                else:
                    scan_not_inf.append(3.5)
            #print(scan_not_inf)
            x = np.asarray(scan_not_inf)
            x_tensor = torch.FloatTensor(x)
            y = self.model(x_tensor)
            self.y = y
            #self.context_type = y.argmax().item()
            context_type = y.argmax().item()
            #print context_type
            self.history.append(context_type)
            past = np.array(self.history[max(0, len(self.history)-3):])
            #print(past)
            self.context_type = stats.mode(past)[0][0]
            self.context_type = context_type
    
    def get_context_type(self):
        context_type = self.context_type
        return context_type

    def get_y(self):
        y = self.y
        y = F.softmax(y)
        return y
#####################################################3


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
                state   = torch.cuda.FloatTensor(state).unsqueeze(0)
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

    def action_to_params_appld(self, action):

        #Parameters appld per context
        if action == 0 :
            params = { 'max_vel_x' : 1.59, 'min_vel_x' : -1.59, 'max_vel_trans': 1.59, 'max_vel_theta': 0.89, 'vx_samples': 12, 'vth_samples': 18, 'occdist_scale': 0.4,
            'path_distance_bias': 16, 'goal_distance_bias': 7, 'inflation_radius': 0.42}
        elif action == 1:
            params = { 'max_vel_x' : 0.8, 'min_vel_x' : -0.8, 'max_vel_trans': 0.8, 'max_vel_theta': 0.73, 'vx_samples': 6, 'vth_samples': 42, 'occdist_scale': 0.04,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.4}
        elif action == 2:
            params = { 'max_vel_x' : 0.71, 'min_vel_x' : -0.71, 'max_vel_trans': 0.71, 'max_vel_theta': 0.91, 'vx_samples': 16, 'vth_samples': 53, 'occdist_scale': 0.55,
            'path_distance_bias': 16, 'goal_distance_bias': 18, 'inflation_radius': 0.39}
        elif action == 3:
            params = { 'max_vel_x' : 0.25, 'min_vel_x' : -0.25, 'max_vel_trans': 0.25, 'max_vel_theta': 1.34, 'vx_samples': 8, 'vth_samples': 59, 'occdist_scale': 0.43,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.40}

        return params

    def action_to_params_appld_extended(self, action):

        #Parameters appld per context
        if action == 0 :
            params = { 'max_vel_x' : 1.59, 'min_vel_x' : -1.59, 'max_vel_trans': 1.59, 'max_vel_theta': 0.89, 'vx_samples': 12, 'vth_samples': 18, 'occdist_scale': 0.4,
            'path_distance_bias': 16, 'goal_distance_bias': 7, 'inflation_radius': 0.42}
        elif action == 1:
            params = { 'max_vel_x' : 0.8, 'min_vel_x' : -0.8, 'max_vel_trans': 0.8, 'max_vel_theta': 0.73, 'vx_samples': 6, 'vth_samples': 42, 'occdist_scale': 0.04,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.4}
        elif action == 2:
            params = { 'max_vel_x' : 0.71, 'min_vel_x' : -0.71, 'max_vel_trans': 0.71, 'max_vel_theta': 0.91, 'vx_samples': 16, 'vth_samples': 53, 'occdist_scale': 0.55,
            'path_distance_bias': 16, 'goal_distance_bias': 18, 'inflation_radius': 0.39}
        elif action == 3:
            params = { 'max_vel_x' : 0.25, 'min_vel_x' : -0.25, 'max_vel_trans': 0.25, 'max_vel_theta': 1.34, 'vx_samples': 8, 'vth_samples': 59, 'occdist_scale': 0.43,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.40}
        if action == 4 :
            params = { 'max_vel_x' : 0.15, 'min_vel_x' : -0.15, 'max_vel_trans': 0.15, 'max_vel_theta': 1.34, 'vx_samples': 8, 'vth_samples': 59, 'occdist_scale': 0.43,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.15}
        elif action == 5:
            params = { 'max_vel_x' : 1, 'min_vel_x' : -1, 'max_vel_trans': 1, 'max_vel_theta': 0.73, 'vx_samples': 6, 'vth_samples': 42, 'occdist_scale': 0.04,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.35}
        elif action == 6:
            params = { 'max_vel_x' : 0.5, 'min_vel_x' : -0.5, 'max_vel_trans': 0.5, 'max_vel_theta': 0.91, 'vx_samples': 16, 'vth_samples': 53, 'occdist_scale': 0.55,
            'path_distance_bias': 16, 'goal_distance_bias': 18, 'inflation_radius': 0.3}
        elif action == 7:
            params = { 'max_vel_x' : 0.2, 'min_vel_x' : -0.2, 'max_vel_trans': 0.2, 'max_vel_theta': 1.2, 'vx_samples': 8, 'vth_samples': 59, 'occdist_scale': 0.43,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.6}

        return params

    def test(self):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :return:
        """

        # Variables
        global test_file

        num_steps = 6000000

        max_time_episode = 2000

        test_file +=  ".csv"
        max_episodes = 70
        trajectory = 0
        #######################

        self.update_target()

        ### Context predictor ###
        predictor = Predictor()
        predictor.context_type = 0


        for i in range(max_episodes):
            print(f"Episode num: {i}")
            state = self.env.reset()

            #### Start score parameters ###
            episode_steps, episode_reward, abort, done, score = start_score_per_episode()
            ###############################

            while not done:

                ### Context prediction ###
                scan = self.env.get_scan()
                predictor.update_context(scan)
                context_type = predictor.get_context_type()

                action = self.current_model.act(state, 0)
                if torch.is_tensor(action):
                    action = action.item()
                #if action ==3:
                #    print(f"Context is: {environments[context_type]}")
                #    print(f"Action is {action}")
                next_state, reward, done = self.env.step_vae_mdrnn(self.action_to_params_appld(action))
                #next_state, reward, done = self.env.step([1, -1, 0, 0.5, 0.5, 0.5, 0.5])
                state = next_state

                ### Update score ###
                episode_steps, episode_reward, abort, done, score = update_score_during_episode(episode_steps, episode_reward, abort, done, score, reward, context_type,
                 self.env)
                ####################
            

            ### Update score end of episode ###
            score = update_score_finished_episode(score, abort, trajectory, episode_steps)
            ####################################

            ### Write score ###
            write_score(score)
            ###################

            #read_score()
            
            ### Update trajectory ###
            trajectory += 1
            trajectory = trajectory % 7
            #########################
            
            print(f"Timesteps: {episode_steps}, Total reward: {episode_reward}")

########### Score functions #################
def start_score_per_episode():
    episode_steps = 0
    episode_reward = 0
    abort = 0
    done = False
    score = [None]*7
    score[1] = []
    score[4] = []
    score[5] = []
    score[6] = []

    return episode_steps, episode_reward, abort, done, score


def write_score(score):
    # Write results to test file
    global test_file

    my_file = Path(test_file)
    file_exists = my_file.is_file()
    if not file_exists:
        with open(test_file, 'a', newline='') as csvfile:
            fieldnames = ['trajectory', 'context','success', 'time', 'min_dist', 'min_vel', 'rob_vel']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'trajectory': score[0], 'context': score[1], 'success': score[2], 'time': score[3], 'min_dist': score[4], 
            'min_vel': score[5], 'rob_vel': score[6]})
    else:
        with open(test_file, 'a', newline='') as csvfile:
            fieldnames = ['trajectory', 'context','success', 'time', 'min_dist', 'min_vel', 'rob_vel']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'trajectory': score[0], 'context': score[1], 'success': score[2], 'time': score[3], 'min_dist': score[4], 
            'min_vel': score[5], 'rob_vel': score[6]})

def read_score():
    # Read results from test file
    global test_file

    with open(test_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['trajectory'], row['context'], row['success'], row['time'], row['min_dist'], row['min_vel'], row['rob_vel'])

def score_sensors(min_scan_bufferr, min_scan_buffer_sizee, odom_buferr, odom_buffer_sizee):
    min_dist_list = []
    min_vel_list = []
    rob_vel_list = []
    for itr in range(1, min(10, min_scan_buffer_sizee+1)):
       rob_vel_list.append(odom_buferr[-itr])
       min_dist_list.append(min_scan_bufferr[-itr])
       min_vel_list.append(min_scan_bufferr[-itr] - min_scan_bufferr[-itr -1])
    return min_dist_list, min_vel_list, rob_vel_list

def update_score_during_episode(episode_steps, episode_reward, abort, done, score, reward, context_type, env):
    if reward < -40:
        print("Aborted")
        abort = 1
    min_dist_list, min_vel_list, rob_vel_list = score_sensors(env.get_min_scan_buffer(), env.get_min_scan_buffer_size(), env.get_odom_buffer()
    ,env.get_odom_buffer_size())
    for i in range(len(min_dist_list)):
        #if min_dist_list[i] < 1.5:
        score[1].append(context_type)
        score[4].append(min_dist_list[i])
        score[5].append(min_vel_list[i])
        score[6].append(rob_vel_list[i])
    episode_steps += 1
    episode_reward += reward

    # Failure if time is more than 400 timesteps
    if episode_steps >2000:
        abort = 1
        done = 1
    
    return episode_steps, episode_reward, abort, done, score

def update_score_finished_episode(score, abort, trajectory, episode_steps):
    score[0] = trajectory
    score[2] = 1 - abort
    score[3] = episode_steps/3

    return score

##################################################

def main():

    current_model = DQN(num_obs, num_actions)
    target_model  = DQN(num_obs, num_actions)
    replay_buffer = NaivePrioritizedBuffer(10000)
    env = task_env.PtdrlTaskEnv()


    print("Loading prediction network")
    current_model.load_state_dict(torch.load(dir_path + '/checkpoint_a_0.pt'))
    current_model.eval()

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
    
    complete_model = DDQN_Replay(num_obs, num_actions, current_model, target_model, replay_buffer, env)
    
    complete_model.test()
 

if __name__ == '__main__':
    rospy.init_node('init_test')
    main()
