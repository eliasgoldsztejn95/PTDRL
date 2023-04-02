#!/usr/bin/env python2

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from numpy import inf
from scipy import stats, interpolate
import time
import matplotlib.pyplot as plt 

from models.vae import VAE
from models.mdrnn import MDRNNCell
from PIL import Image
from pathlib import Path
import csv



import rospy
import copy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import  Twist
import dynamic_reconfigure.client
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest

USE_CUDA = torch.cuda.is_available()

# Environment variables
num_obs = 290
num_actions = 8

dir_path = os.path.dirname(os.path.realpath(__file__))

# Constants
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

##### VAE and MDRNN networks #####
environments = {0:"open", 1:"door", 2:"curve", 3:"obstacles"}

# Folders
dir_path_context = dir_path + "/networks/context_door.ckpt"
dir_path_vae_input_inflation = dir_path + "/networks/vae_input_inflation/best.tar"
dir_path_mdrnn = dir_path + "/networks/mdnrnn/best.tar"

global test_file
test_file = dir_path + "/scores" + "/score_ptdrl_obstacles_6"

algo = 'PTDRL2'
context = 3
print(f"Context is: {environments[context]}")


class RobEnv():
    def __init__(self):
        self.rate_rob = rospy.Rate(10)
        self.odomm = Odometry()
        self.costmapp = OccupancyGrid()
        self.cmd_vell = Twist()
        self.scann = LaserScan()

        # Sensors
        print("Suscribing to scan")
        rospy.Subscriber("scan", LaserScan, self._scan_callback)
        print("Suscribed!")

        print("Suscribing to odom")
        rospy.Subscriber("odom", Odometry, self._odom_callback)
        print("Suscribed!")

        # Local planner
        print("Suscribing to cmd_vel")
        rospy.Subscriber('/cmd_vel', Twist, self._cmd_vel_callback)
        print("Suscribed!")

        # Costmap
        print("Suscribing to costmap")
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self._costmap_callback)
        print("Suscribed!")

        print("Waiting for move_base server")
        rospy.wait_for_service('/move_base/clear_costmaps')
        self._clear_costmap_service = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        print("Connected!")

        # Actions
        self._tuning_client_local = dynamic_reconfigure.client.Client('move_base/DWAPlannerROS',timeout=4, config_callback=None)
        #self._tuning_client_local = dynamic_reconfigure.client.Client('move_base/TebLocalPlannerROS',timeout=4, config_callback=None)
        self._tuning_client_inflation = dynamic_reconfigure.client.Client('move_base/local_costmap/inflation_layer',timeout=4, config_callback=None)

        super(RobEnv, self).__init__()

    def _scan_callback(self, msg):

        # Update lidar reading
        current_time = rospy.Time.now()
        self.scann.header.stamp = current_time
        self.scann.header.frame_id = msg.header.frame_id 
        self.scann.angle_min = msg.angle_min
        self.scann.angle_max = msg.angle_max
        self.scann.angle_increment = msg.angle_increment
        self.scann.time_increment = msg.time_increment
        self.scann.range_min = msg.range_min
        self.scann.range_max = msg.range_max
        self.scann.ranges = msg.ranges
        self.scann.intensities = msg.intensities

    def _odom_callback(self, msg):

        # Update odometry reading
        current_time = rospy.Time.now()
        self.odomm.header.stamp = current_time
        self.odomm.pose.pose.position.x = msg.pose.pose.position.x
        self.odomm.pose.pose.position.y = msg.pose.pose.position.y
        self.odomm.pose.pose.orientation.x = msg.pose.pose.orientation.x
        self.odomm.pose.pose.orientation.y = msg.pose.pose.orientation.y
        self.odomm.pose.pose.orientation.z = msg.pose.pose.orientation.z
        self.odomm.pose.pose.orientation.w = msg.pose.pose.orientation.w
        self.odomm.twist.twist.linear.x = msg.twist.twist.linear.x
        self.odomm.twist.twist.linear.y = msg.twist.twist.linear.y
        self.odomm.twist.twist.linear.z = msg.twist.twist.linear.z
        self.odomm.twist.twist.angular.x = msg.twist.twist.angular.x
        self.odomm.twist.twist.angular.y = msg.twist.twist.angular.y
        self.odomm.twist.twist.angular.z = msg.twist.twist.angular.z

    def _costmap_callback(self, msg):
        current_time = rospy.Time.now()
        self.costmapp.header.stamp = current_time
        self.costmapp.header.frame_id = msg.header.frame_id
        self.costmapp.info.height = msg.info.height
        self.costmapp.info.width = msg.info.width
        self.costmapp.info.origin.position.x = msg.info.origin.position.x
        self.costmapp.info.origin.position.y = msg.info.origin.position.y
        self.costmapp.info.origin.orientation.x = msg.info.origin.orientation.x 
        self.costmapp.info.origin.orientation.y = msg.info.origin.orientation.y 
        self.costmapp.info.origin.orientation.z = msg.info.origin.orientation.z 
        self.costmapp.info.origin.orientation.w = msg.info.origin.orientation.w

        self.costmapp.data = msg.data

    def _cmd_vel_callback(self, msg):
        self.cmd_vell.linear.x = msg.linear.x
        self.cmd_vell.angular.z = msg.angular.z

    def clear_costmap(self):
        # Clear costmaps

        clear_costmap_object = EmptyRequest()

        result = self._clear_costmap_service(clear_costmap_object)
        #print("Clearing costmaps" + str(result))

    def tune_parameters(self, params):
        
        inflation = False
        inflation_param = {'inflation_radius': 1}

        if 'inflation_radius' in params.keys():
            inflation = True
            inflation_param['inflation_radius'] = params['inflation_radius']

        local_params = copy.deepcopy(params)
        if inflation:
            del local_params['inflation_radius']

        self._tuning_client_local.update_configuration(local_params)
        #print(cg)
        if inflation:
            self._tuning_client_inflation.update_configuration(inflation_param)

    def get_odom(self):
        odom = Odometry()
        current_time = rospy.Time.now()
        odom.header.stamp = current_time
        odom.pose.pose.position.x = self.odomm.pose.pose.position.x
        odom.pose.pose.position.y = self.odomm.pose.pose.position.y
        odom.pose.pose.orientation.x = self.odomm.pose.pose.orientation.x
        odom.pose.pose.orientation.y = self.odomm.pose.pose.orientation.y
        odom.pose.pose.orientation.z = self.odomm.pose.pose.orientation.z
        odom.pose.pose.orientation.w = self.odomm.pose.pose.orientation.w
        odom.twist.twist.linear.x = self.odomm.twist.twist.linear.x
        odom.twist.twist.linear.y = self.odomm.twist.twist.linear.y
        odom.twist.twist.linear.z = self.odomm.twist.twist.linear.z
        odom.twist.twist.angular.x = self.odomm.twist.twist.angular.x 
        odom.twist.twist.angular.y = self.odomm.twist.twist.angular.y 
        odom.twist.twist.angular.z = self.odomm.twist.twist.angular.z 
        return odom
    
    def get_scan(self):
        scan = LaserScan()
        current_time = rospy.Time.now()
        scan.header.stamp = current_time
        scan.header.frame_id = self.scann.header.frame_id 
        scan.angle_min = self.scann.angle_min
        scan.angle_max = self.scann.angle_max
        scan.angle_increment = self.scann.angle_increment
        scan.time_increment = self.scann.time_increment
        scan.range_min = self.scann.range_min
        scan.range_max = self.scann.range_max
        scan.ranges = self.scann.ranges
        scan.intensities = self.scann.intensities

        return scan
    
    def get_costmap(self):
        costmap = OccupancyGrid()

        current_time = rospy.Time.now()
        costmap.header.stamp = current_time
        costmap.header.frame_id = self.costmapp.header.frame_id
        costmap.info.height = self.costmapp.info.height
        costmap.info.width = self.costmapp.info.width
        costmap.info.origin.position.x = self.costmapp.info.origin.position.x
        costmap.info.origin.position.y = self.costmapp.info.origin.position.y
        costmap.info.origin.orientation.x = self.costmapp.info.origin.orientation.x 
        costmap.info.origin.orientation.y = self.costmapp.info.origin.orientation.y 
        costmap.info.origin.orientation.z = self.costmapp.info.origin.orientation.z 
        costmap.info.origin.orientation.w = self.costmapp.info.origin.orientation.w

        costmap.data = self.costmapp.data

        return costmap
    
    def get_cmd_vel(self):
        cmd_vel = copy.deepcopy(self.cmd_vell)

        return cmd_vel

class TaskEnv(RobEnv):

    """ Class for PTDRL Task environment
    """
    def __init__(self):
        # NN'S
        self.img_channels = 3
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.vae_input_inflation = VAE(self.img_channels, 32).to(self.device)
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device)

        self.state_input_inflation = torch.load(dir_path_vae_input_inflation)
        self.vae_input_inflation.load_state_dict(self.state_input_inflation['state_dict'])

        self.state_mdrnn = torch.load(dir_path_mdrnn)
        self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in self.state_mdrnn['state_dict'].items()})

        # INitialize netowrk variables
        self.hidden =[torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
        # Robot
        self.rate_task = rospy.Rate(10)

        self.costmap_counter = 0
        self.clear_costmap_counter = 0
        self.costmap_data = None

        RobEnv.__init__(self)



    def step_vae_mdrnn(self, action, dist):
        # Step for vae and mdrnn
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 5:
            self.clear_costmap()
            self.clear_costmap_counter = 0
        
        # Update hidden
        obs_costmap = self.get_costmap()
        self.costmap_data = obs_costmap.data
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap, dist)
        mdrnn_action = self.get_action_for_mdrnn(action)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)

        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap, dist)
        vel = self.get_vel()
        obs = self.extract_3(latent_mu, self.hidden[0], vel)
        reward = self.reward_func()


        return obs, reward

    def reward_func(self):
        # Robot velocity
        odom = self.get_odom()
        vel_rob = np.abs(odom.twist.twist.linear.x)
        scan = self.get_scan()
        min_range = np.min(scan.ranges)

        reward_func = vel_rob*(-1 if min_range < 0.75 else 1) - 1

        return reward_func

    def process_obs_3(self, obs, dist):
        # Encode costmap/s.
        image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)

        for itr in range(self.img_channels):
            image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs)
          
        # Encode input image
        image_i = torch.tensor(image_i)
        image_i = image_i.to(self.device, dtype=torch.float)
        image_i_o, mu_i_o, logvar_i_o = self.vae_input_inflation(image_i)


        ####################################################
        #Show images #######################################
        self.costmap_counter += 1

        if self.costmap_counter %5 == 0:

            fig2, (ax11, ax22, ax33) = plt.subplots(1, 3,figsize=(16, 4), dpi=80)
            fig2.suptitle('Images')

            im1 = Image.fromarray(np.fliplr(np.squeeze((np.divide(np.array_split(obs.data, 60),100))))*256)
            imgplot = ax11.imshow(im1, aspect="auto")
            ax11.set_title("Original Cost-map")


            im2 =  Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
            imgplot = ax22.imshow(im2, aspect="auto")
            ax22.set_title("Reconstructed Cost-map")

            actions = list(np.arange(0, 8, 1, dtype=int))
            print(f"len actions {actions}")
            ax33.bar(actions, softmax(dist), color ='blue')
            
            ax33.set_xlabel("Actions")
            ax33.set_ylabel("Normalized Q_values")
            ax33.set_ylim(0,1)

            #im3 =  Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
            #imgplot = ax33.imshow(im3)
            ax33.set_title("Action distribution")

        ####################################################
        ####################################################
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        
        return mu_i_o, logvar_i_o

    def extract_3(self, mu_i_o, logvar_i_o, vel):
        mu_i_o = mu_i_o.cpu().data.numpy()[0]
        logvar_i_o = logvar_i_o.cpu().data.numpy()[0]

        obs_cat = np.concatenate((mu_i_o, logvar_i_o, vel), axis = 0)

        return (obs_cat)

    def costmap_to_np(self, costmap):
        # Occupancy grid to numpy
        costmap_np = np.zeros([60,60])
        if costmap is not None:
            costmap_np = np.divide(np.array_split(costmap.data, 60),100) # costmap size is 60
        return costmap_np

    def get_action_for_mdrnn(self, action):
        inflation = action['inflation_radius']
        cmd_vel = self.get_cmd_vel()
        mdrnn_action = np.zeros([1,3])
        mdrnn_action[0,0] = cmd_vel.linear.x
        mdrnn_action[0,1] = cmd_vel.angular.z
        mdrnn_action[0,2] = inflation

        return torch.cuda.FloatTensor(mdrnn_action)

    def get_vel(self):
        odom = self.get_odom()
        vel = np.zeros([2])
        vel[0] = odom.twist.twist.linear.x
        vel[1] = odom.twist.twist.angular.z
        
        return vel


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
            x = np.arange(0, 560)
            f = interpolate.interp1d(x, np.array(scan_not_inf))
            x_new = np.arange(0, 559, 559/720)
            scan_not_inf = list(f(x_new))
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

########### DDQN ##################################3##

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
            dist = q_value.data[0]
        else:
            action = random.randrange(self.num_actions)
        return action, dist

class DDQN_Replay():
    def __init__(self, num_obs, num_actions, current_model, target_model, replay_buffer, task):

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.current_model = current_model
        self.target_model = current_model
        self.replay_buffer = replay_buffer
        self.TaskEnv = task
        time.sleep(1)

        super(DDQN_Replay, self).__init__()
    
    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())
    

    def action_to_params_ddqn(self, action):

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

    def action_to_params_appld(self, action):

        #Parameters appld per context
        if action == 0 :
            params = { 'max_vel_x' : 1.59, 'min_vel_x' : -1.59, 'max_vel_trans': 1.59, 'max_vel_theta': 0.89, 'vx_samples': 12, 'vth_samples': 18, 'occdist_scale': 0.4,
            'path_distance_bias': 16, 'goal_distance_bias': 7, 'inflation_radius': 0.42}
        elif action == 1:
            params = { 'max_vel_x' : 0.25, 'min_vel_x' : -0.25, 'max_vel_trans': 0.25, 'max_vel_theta': 1.34, 'vx_samples': 8, 'vth_samples': 59, 'occdist_scale': 0.43,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.40}
        elif action == 2:
            params = { 'max_vel_x' : 0.8, 'min_vel_x' : -0.8, 'max_vel_trans': 0.8, 'max_vel_theta': 0.73, 'vx_samples': 6, 'vth_samples': 42, 'occdist_scale': 0.04,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.4}
        elif action == 3:
            params = { 'max_vel_x' : 0.71, 'min_vel_x' : -0.71, 'max_vel_trans': 0.71, 'max_vel_theta': 0.91, 'vx_samples': 16, 'vth_samples': 53, 'occdist_scale': 0.55,
            'path_distance_bias': 16, 'goal_distance_bias': 18, 'inflation_radius': 0.39}

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
            params = { 'max_vel_x' : 1.2, 'min_vel_x' : -1.2, 'max_vel_trans': 1.2, 'max_vel_theta': 1.2, 'vx_samples': 8, 'vth_samples': 59, 'occdist_scale': 0.43,
            'path_distance_bias': 32, 'goal_distance_bias': 20, 'inflation_radius': 0.3}

        return params

    def action_to_params_default(self):
        params = { 'max_vel_x' : 0.5, 'min_vel_x' : -0.5, 'max_vel_trans': 0.5, 'max_vel_theta': 1.57, 'vx_samples': 6, 'vth_samples': 20, 'occdist_scale': 0.1,
            'path_distance_bias': 16, 'goal_distance_bias': 7, 'inflation_radius': 0.42}
        
        return params

    def test(self):
        """
        trains the model's parameters over a fixed number of epochs, specified by `n_epochs`, as long as the loss keeps decreasing.
        :param dataset: `Dataset` object
        :return:
        """

        self.update_target()

        ### Context predictor ###
        predictor = Predictor()
        predictor.context_type = 0
        state = np.zeros([290])

        #### Start score parameters ###
        episode_steps, episode_reward, abort, done, score = start_score_per_episode()
        ###############################
        time.sleep(5)


        while True:
            ### Context prediction ###
            scan = self.TaskEnv.get_scan()
            predictor.update_context(scan)
            context_type = predictor.get_context_type()

            action, dist = self.current_model.act(state, 0)
            if torch.is_tensor(action):
                action = action.item()
                dist = dist.tolist()
            print(dist)
            #if action ==3:
            #print(f"Context is: {environments[context_type]}")
            print(f"Action is {action}")
            ### SET ALGORITHM ###
            if algo == 'DEFAULT':
                params = self.action_to_params_default()
            elif algo == 'APPLD':
                params = self.action_to_params_appld(context)
            elif algo == 'PTDRL1':
                params = self.action_to_params_ddqn(action)
            elif algo == 'PTDRL2':
                params = self.action_to_params_appld_extended(action)

            next_state, reward = self.TaskEnv.step_vae_mdrnn(params, dist)
            state = next_state

            ### Update score ###
            episode_steps, episode_reward, abort, done, score = update_score_during_episode(episode_steps, episode_reward, abort, done, score, reward, context_type,self.TaskEnv)
            ####################

            ### Update score end of episode ###
            score = update_score_finished_episode(score, abort, 0, episode_steps)
            ####################################

            ### Write score ###
            write_score(score)
            ###################
            
            print(f"Timesteps: {episode_steps}, Current total reward: {episode_reward}")

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
        with open(test_file, 'w', newline='') as csvfile:
            fieldnames = ['trajectory', 'context','success', 'time', 'min_dist', 'min_vel', 'rob_vel']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'trajectory': score[0], 'context': score[1], 'success': score[2], 'time': score[3], 'min_dist': score[4], 
            'min_vel': score[5], 'rob_vel': score[6]})
    else:
        with open(test_file, 'w', newline='') as csvfile:
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

def score_sensors(scan, odom):
    min_dist = min(scan.ranges)
    rob_vel = np.abs(odom.twist.twist.linear.x)

    return min_dist, 0, rob_vel

def update_score_during_episode(episode_steps, episode_reward, abort, done, score, reward, context_type, task):
    if reward < -40:
        print("Aborted")
        abort = 1
    min_dist, min_vel, rob_vel = score_sensors(task.get_scan(), task.get_odom())
    score[1].append(context_type)
    score[4].append(min_dist)
    score[5].append(min_vel)
    score[6].append(rob_vel)
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
    score[3] = episode_steps

    return score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

##################################################


def main():
    global num_actions

    current_model = DQN(num_obs, num_actions)
    target_model  = DQN(num_obs, num_actions)
    replay_buffer = NaivePrioritizedBuffer(10000)
    task = TaskEnv()

    print("Loading prediction network")
    if algo == 'PTDRL2':
        num_actions = 8
        current_model.load_state_dict(torch.load(dir_path + '/checkpoint_dqn_mdrnn_e_3_best.pt'))
        current_model.eval()
    else:
        num_actions = 4
        current_model.load_state_dict(torch.load(dir_path + '/checkpoint_a_0.pt'))
        current_model.eval()

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
    
    complete_model = DDQN_Replay(num_obs, num_actions, current_model, target_model, replay_buffer, task)
    
    complete_model.test()
 

if __name__ == '__main__':
    rospy.init_node('init_test')
    main()
