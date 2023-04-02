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

from IPython.display import clear_output
import matplotlib.pyplot as plt

import rospy
import task_env

USE_CUDA = torch.cuda.is_available()

# Environment variables
num_obs = 34# Lidar: 722, VAE: 34, VAE_MDRNN 290
num_actions = 8

dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.env = env

        super(DDQN_Replay, self).__init__()
    
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def compute_td_loss(self, batch_size, beta, optimizer, gamma):
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta) 

        with torch.no_grad():
            state      = torch.cuda.FloatTensor(np.float32(state))
            next_state = torch.cuda.FloatTensor(np.float32(next_state))
            action     = torch.cuda.LongTensor(action)
            reward     = torch.cuda.FloatTensor(reward)
            done       = torch.cuda.FloatTensor(done)
            weights    = torch.cuda.FloatTensor(weights)

        q_values      = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        loss  = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss  = loss.mean()
            
        optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        optimizer.step()
        
        return loss

    def save_checkpoint(self, current_best):
        '''Saves model when validation loss decrease.'''
        #print(f'Validation loss decreased ({val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(self.current_model.state_dict(), dir_path + '/checkpoint_dqn_mdrnn_e_0_v_' + current_best + '.pt')

    
    def plot(self, frame_idx, rewards, losses):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()
    
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
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.6} # 1.2 -> 0.2

        return params
    
    def action_to_params_genetic(self, action):

        # Parameters GA per context: the rest of contexts have the parameters predicted by APPLD
        params = None
        if action == 0 :
            params_g = [ 1.5733548 ,  1.0709816 , 16.17663   , 21.928034  ,  0.40977708,
        13.26083   ,  7.837092  ,  0.4232008 ]

            params = { 'max_vel_x' : params_g[0], 'min_vel_x' : -params_g[0], 'max_vel_trans': params_g[0], 'max_vel_theta': params_g[1], 
            'vx_samples': params_g[2], 'vth_samples': params_g[3], 'occdist_scale': params_g[4], 'path_distance_bias': params_g[5], 
            'goal_distance_bias': params_g[6], 'inflation_radius': params_g[7]}

        elif action == 1:
            params_g = [ 0.16455802,  1.469773  , 10.114808  , 51.46517   ,  0.3278064 ,
        19.416002  , 19.857208  ,  0.4470366 ]

            params = { 'max_vel_x' : params_g[0], 'min_vel_x' : -params_g[0], 'max_vel_trans': params_g[0], 'max_vel_theta': params_g[1], 
            'vx_samples': params_g[2], 'vth_samples': params_g[3], 'occdist_scale': params_g[4], 'path_distance_bias': params_g[5], 
            'goal_distance_bias': params_g[6], 'inflation_radius': params_g[7]}

        elif action == 2:
            params_g = [ 0.76337403,  0.75673604,  7.904575  , 45.27914   ,  0.13230471,
        31.055052  , 20.818293  ,  0.296872  ]

            params = { 'max_vel_x' : params_g[0], 'min_vel_x' : -params_g[0], 'max_vel_trans': params_g[0], 'max_vel_theta': params_g[1], 
            'vx_samples': params_g[2], 'vth_samples': params_g[3], 'occdist_scale': params_g[4], 'path_distance_bias': params_g[5], 
            'goal_distance_bias': params_g[6], 'inflation_radius': params_g[7]}

        elif action == 3:
            params_g = [ 0.68968034,  0.82407176, 14.584932  , 48.501217  ,  0.47780252,
        19.956453  , 20.505772  ,  0.44962054]

            params = { 'max_vel_x' : params_g[0], 'min_vel_x' : -params_g[0], 'max_vel_trans': params_g[0], 'max_vel_theta': params_g[1], 
            'vx_samples': params_g[2], 'vth_samples': params_g[3], 'occdist_scale': params_g[4], 'path_distance_bias': params_g[5], 
            'goal_distance_bias': params_g[6], 'inflation_radius': params_g[7]}

        return params

    def action_to_params_teb(self, action):

        #Parameters teb per context
        if action == 0 :
            params = { 'max_vel_x' : 1.5, 'max_vel_theta': 0.9, 'dt_ref': 0.1, 'weight_max_vel_x': 3, 'weight_obstacle': 50,
            'weight_viapoint': 2, 'weight_max_vel_theta': 3, 'inflation_radius': 0.42}
        elif action == 1:
            params = { 'max_vel_x' : 0.8, 'max_vel_theta': 0.7, 'dt_ref': 0.2, 'weight_max_vel_x': 2, 'weight_obstacle': 50,
            'weight_viapoint': 1.5, 'weight_max_vel_theta': 2, 'inflation_radius': 0.4}
        elif action == 2:
            params = { 'max_vel_x' : 0.5, 'max_vel_theta': 0.9, 'dt_ref': 0.3, 'weight_max_vel_x': 1, 'weight_obstacle': 60,
            'weight_viapoint': 1, 'weight_max_vel_theta': 1, 'inflation_radius': 0.39}
        elif action == 3:
            params = { 'max_vel_x' : 0.2,  'max_vel_theta': 1.3, 'dt_ref': 0.5, 'weight_max_vel_x': 2, 'weight_obstacle': 70,
            'weight_viapoint': 0.5, 'weight_max_vel_theta': 1, 'inflation_radius': 0.40}

        return params
    
    def test_network(self, num_episodes):
        # Initialize total_reward
        total_reward = 0
        for i in range(num_episodes):
            # Reset env
            state = self.env.reset()
            done = False
            while not done:
                action = self.current_model.act(state, 0)
                if torch.is_tensor(action):
                    action = action.item()
                #print(f"Action is {action}")
                next_state, reward, done = self.env.step_vae_mdrnn(self.action_to_params_appld_extended(action))
                
                state = next_state
                total_reward += reward

        total_reward = total_reward/num_episodes
        if total_reward >  self.best_rewards:
            self.best_rewards = total_reward
            print(f"Best reward is {total_reward}")
            self.save_checkpoint("best")


    def fit(self):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :return:
        """

        # Learning variables

        num_steps = 6000000
        batch_size = 32
        gamma      = 0.99

        beta_start = 0.4
        beta_frames = 1000  #1000
        beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500 #500

        epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

        max_episode = 2501
        max_time_episode = 2000
        #######################

        optimizer = optim.Adam(self.current_model.parameters())

        min_loss = 1
        loss = 1
        all_losses = []

        all_rewards = []
        episode_reward = 0

        self.update_target()

        state = self.env.reset()

        episode = 1
        time_episode = 0

        f_r = open(dir_path + "/rewards_vae_mdrnn_e_0_v.txt", "w")
        f_l = open(dir_path + "/losses_vae_mdrnn_e_0_v.txt", "w")
        epsilon = epsilon_by_frame(episode)
        beta = beta_by_frame(episode)
        self.best_rewards = -100000

        for frame_idx in range(1, num_steps + 1):
            #epsilon = epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)
            if torch.is_tensor(action):
                action = action.item()
            #print(f"Action is {action}")
            next_state, reward, done = self.env.step_vae_mdrnn(self.action_to_params_appld_extended(action))
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            time_episode += 1
            
            if time_episode >= max_time_episode:
                print("Max time!!!")
                done = 1
                episode_reward += -1000
            
            if done:
                print(f"Episode: {episode} - Episode time: {time_episode} - Rewards: {episode_reward}")
                episode += 1
                state = self.env.reset()
                all_rewards.append(episode_reward)
                a = np.array(all_rewards)
                with open(dir_path + "/rewards_vae_mdrnn_e_0_v.txt", "w") as f_r:
                    np.savetxt(f_r, a, fmt='%1.3f', newline=", ")

                all_losses.append(loss.cpu().data.numpy())
                b = np.array(all_losses)
                with open(dir_path + "/losses_vae_mdrnn_e_0_v.txt", "w") as f_l:
                    np.savetxt(f_l, b, fmt='%1.3f', newline=", ")

                episode_reward = 0
                time_episode = 0

                epsilon = epsilon_by_frame(episode)
                beta = beta_by_frame(episode)
                print(f"epsilon is {epsilon}")
                print(f"beta is {beta}")

                
            if len(self.replay_buffer) > batch_size:
                #print("Computing loss")
                #print(f"loss is {loss}")
                #beta = beta_by_frame(frame_idx)
                loss = self.compute_td_loss(batch_size, beta, optimizer, gamma)
                
            if frame_idx % 50 == 0:
                #self.plot(frame_idx, all_rewards, losses)
                #print("Saving network")
                #print(f"loss is {loss}")
                if loss < min_loss:
                    min_loss = loss
                self.save_checkpoint("current")

            if done and episode % 300 == 0:
                print(f"Starting test for episode: {episode}")
                self.test_network(28)
                state = self.env.reset()
                print("Finished test")
                
            if frame_idx % 1000 == 0:
                #print("Updating target")
                self.update_target()
            
            if episode >= max_episode:
                break
            

        return min_loss


def main():

    current_model = DQN(num_obs, num_actions)
    target_model  = DQN(num_obs, num_actions)
    replay_buffer = NaivePrioritizedBuffer(100000)
    env = task_env.PtdrlTaskEnv()

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
    
    complete_model = DDQN_Replay(num_obs, num_actions, current_model, target_model, replay_buffer, env)
    
    complete_model.fit()
 

if __name__ == '__main__':
    rospy.init_node('init_training')
    main()