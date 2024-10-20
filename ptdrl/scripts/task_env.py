#!/usr/bin/env python3

#######################
## Environment class ##
#######################

# Reset environment
# Compute reward
# Perform action
# Get obs
# Is done


import rospy
import actionlib
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped , PoseStamped, Twist
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback, MoveBaseResult
from actionlib_msgs.msg import GoalStatusArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from actionlib_msgs.msg import GoalID
import csv
import os
from pathlib import Path
from std_srvs.srv import EmptyRequest,  Empty
from visualization_msgs.msg import MarkerArray, Marker
import dynamic_reconfigure.client
import robot_env
import lstm_mdn_simple
import yaml
from yaml.loader import SafeLoader
import torch
import phydnet_predict
from models.vae import VAE
from models.mdrnn import MDRNNCell
from models.reward import REWARD
from models.sft import SFT
import time

import matplotlib.cm as cm
from random import randrange
import matplotlib.animation as animation
from PIL import Image
import time
from matplotlib.animation import FuncAnimation
from numpy import inf


import matplotlib.pyplot as plt

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Paths
dir_path_vae_input_inflation = dir_path + "/networks/vae_input_inflation/best.tar"
dir_path_mdrnn = dir_path + "/networks/mdrnn/best.tar"

dir_path_yaml = dir_path
dir_path_yaml = dir_path_yaml.split("/")
dir_path_yaml = dir_path_yaml[:-2]
dir_path_yaml += ["params"]
dir_path_yaml = '/'.join(dir_path_yaml)

file = "task_params"
yaml_file = "/" + file + ".yaml"

dir_path_yaml += yaml_file

# Constants
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

class PtdrlTaskEnv(robot_env.PtdrlRobEnv):

    """ Superclass for PTDRL Task environment
    """
    def __init__(self):
        """Initializes a new PTDRL Task environment
        """

        self.get_params()

        # Simulation rate
        self.rate_task = rospy.Rate(10) # Use 10 rate. 5 now 3

        # Counters
        self.counter = -1 #1 Counter to change goal and init pose
        self.clear_costmap_counter = 0  
        self.stuck = 0 # if more than 500 seconds passed and the robot did not move then the environment crashed      
        # Position and goals
        self.init_pose = None
        self.goal = None
        # Robot constants
        self.max_vel_robot = 1

        # NN's
        self.img_channels = 3
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # Initialize networks
        self.vae_input_inflation = VAE(self.img_channels, 32).to(self.device)
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device)

        # SFT and Reward
        self.reward_net = REWARD(290, 4).to(self.device)
        self.sft = SFT(290, 4).to(self.device)
        
        # Load networks

        self.state_input_inflation = torch.load(dir_path_vae_input_inflation)
        self.vae_input_inflation.load_state_dict(self.state_input_inflation['state_dict'])

        self.state_mdrnn = torch.load(dir_path_mdrnn)
        self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in self.state_mdrnn['state_dict'].items()})

        # INitialize netowrk variables
        self.hidden =[torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]

        robot_env.PtdrlRobEnv.__init__(self, model_name = self.model_name, amcl = self.amcl, min_dist_to_obstacle = self.min_dist_to_obstacle,
                                    min_dist_to_goal = self.min_dist_to_goal, num_tracks = self.num_tracks, timeout = self.timeout)

    def get_params(self):
        """
        Get parameters from yaml file. These parameters include:
        1) Goals and starting points inside the hospital
        2) Local planner name
        3) Whether to use amcl or not
        4) Robot name
        5) TImeouts
        6) Option to use discrete or continous actions (In PTDRL it is discrete)
        """

        with open(dir_path_yaml, 'r') as f:
            data = list(yaml.load_all(f, Loader=SafeLoader))
        
        # Time
        self.timeout = data[0]["timeout"]

        # Model
        self.model_name = data[0]["model_name"]

        # Navigation
        self.amcl = data[0]["amcl"]

        # Tracking
        self.num_tracks = data[0]["num_tracks"]

        # Actionsctions
        self.discrete_actions = data[0]["discrete_actions"]
        self.list_tune_params = []

        # World
        self.min_dist_to_obstacle = data[0]["min_dist_to_obstacle"]
        self.min_dist_to_goal = data[0]["min_dist_to_goal"]

        self.list_init_pose = []
        self.list_goals = []

        # Local planner
        self.local_planner = data[0]["local_planner"]
        if self.local_planner == "dwa":
            tune_params = "tune_params_dwa"
        else:
            tune_params = "tune_params_teb"

        for i in range(len(data[0][tune_params])):
            one_param = []
            one_param.append(data[0][tune_params][i]["name"])
            one_param.append([data[0][tune_params][i]["min"], data[0][tune_params][i]["max"]])
            self.list_tune_params.append(one_param)
        #print(f"These are the params {self.list_tune_params}")

        for i in range(len(data[0]["list_init_pose"])):
            self.list_init_pose.append(Pose())
            self.list_goals.append(Pose())
            
            #print(data[0]["list_init_pose"][0]["x"])
            self.list_init_pose[i].position.x = data[0]["list_init_pose"][i]["x"]
            self.list_init_pose[i].position.y = data[0]["list_init_pose"][i]["y"]
            self.list_init_pose[i].position.z = 0
            orientation = quaternion_from_euler(0,0,0)
            
            self.list_init_pose[i].orientation.x =  orientation[0]
            self.list_init_pose[i].orientation.y =  orientation[1]
            self.list_init_pose[i].orientation.z =  orientation[2]
            self.list_init_pose[i].orientation.w =  orientation[3]

            self.list_goals[i].position.x = data[0]["list_goals"][i]["x"]
            self.list_goals[i].position.y = data[0]["list_goals"][i]["y"]
    
    def _set_init_pose(self):
        """
        Initiate robot by locating it in one of the places of the hospital.
        """

        self.init_robot(self.init_pose)
        #print(self.init_pose)
        #print("Robot initialized!!!")
    
    def _send_goal(self):
        """
        Send robot to goal (Declare move_base goal).
        """

        #print(self.goal)
        self.send_goal(self.goal)
        #print("Goal sent!!!")
    
    def reset(self):
        """
        Init variables, set robot in position, clear costmap and send goal.
        """

        self.counter += 1
        iter = self.counter % len(self.list_init_pose)

        self.init_pose = self.list_init_pose[iter]
        self.goal = self.list_goals[iter]

        self._init_env_variables()

        self._set_init_pose()

        self.clear_costmap()

        self._send_goal()

        return np.zeros([290]) # Lidar: 722, VAE: 34, VAE_MDRNN: 290
    
    def _init_env_variables(self):

        self.status_move_base = 0
    

    def step_vae_mdrnn(self, action):
        """
        Step for vae and mdrnn
        Update hidden given action, set action,  wait for rewards, return obs and is_done.
        Ablation indicates which input to use.
        """
        # Ablation: Choose to use lidar, VAE, or VAE + MDN-RNN.
        ablation = 2 # 0 Just lidar, 1 just VAE

        # Clear buffers: Clear costmap buffers.
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        # Clear costmap every 0.5 seconds
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Initialization
        is_done = 0
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        # Update MDN-RNN Neural Network. self.hidden[0] is h. 
        obs_costmap = self.get_costmap()
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        mdrnn_action = self.get_action_for_mdrnn(action)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action: Tune move_base parameters.
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations: Get lidar, costmap, current velocity (v), and VAE (z). 
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(latent_mu, self.hidden[0], vel)
        obs_0 = self.extract_lidar(lidar, vel)
        obs_1 = self.extract_vae(latent_mu, vel)

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        if ablation == 0:
            return obs_0, reward, is_done
        elif ablation == 1:
            return obs_1, reward, is_done
        else:
            return obs, reward, is_done

    def extract_3(self, z, h, v):
        """
        Concatenate VAE (z), MDN-RNN (h), and velocity (v).
        """

        z_t = z.cpu().data.numpy()[0]
        h_t = h.cpu().data.numpy()[0]
        v_t = v

        obs_cat = np.concatenate((z_t, h_t, v_t), axis = 0)

        return (obs_cat)
    
    def extract_lidar(self, lidar, v):
        """
        Concatenate lidar and velocity.
        """
        obs_cat = np.concatenate((lidar, v), axis = 0)

        return (obs_cat)

    def extract_vae(self, z, v):
        """
        Concatenate VAE (z) and velocity (v).
        """
        z_t = z.cpu().data.numpy()[0]
        v_t = v

        obs_cat = np.concatenate((z_t, v_t), axis = 0)

        return (obs_cat)

    def costmap_to_np(self, costmap):
        """
        Convert occupancy grid to numpy.
        """

        costmap_np = np.zeros([60,60])
        if costmap is not None:
            costmap_np = np.divide(np.array_split(costmap.data, 60),100) # costmap size is 60
        return costmap_np   
    
    def process_obs_3(self, obs):
        """
        Encode costmap using VAE Neural Netowrk.
        """

        image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)

        for itr in range(self.img_channels):
            image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs)
          
        # Encode input image
        image_i = torch.tensor(image_i)
        image_i = image_i.to(self.device, dtype=torch.float)
        image_i_o, mu_i_o, logvar_i_o = self.vae_input_inflation(image_i)


        ####################################################
        #Show images #######################################

        # fig2, (ax11, ax22) = plt.subplots(1, 2)
        # fig2.suptitle('Images')

        # im1 = Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax11.imshow(im1)
        # ax11.set_title("Input")


        # im3 = Image.fromarray(np.fliplr(np.squeeze(obs.cpu().data.numpy()))*256)
        # imgplot = ax22.imshow(im3)
        # ax22.set_title("Decoded input")

        #####################################################
        #####################################################

        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # plt.show()
        
        return mu_i_o, logvar_i_o
    
    def translate_continous_action(self, values, exact):
        """
        Translate from numbers to values of parameters for parameter tuning in the case of continous actions.
        """

        action = {}

        for itr in range(len(self.list_tune_params)):
            if exact:
                val = values[itr]
            else:
                val = self.get_action(self.list_tune_params[itr][1][0], self.list_tune_params[itr][1][1], values[itr])
            if self.list_tune_params[itr][0] != 'vel':
                action[self.list_tune_params[itr][0]] = val
            else:
                if self.local_planner == "dwa":
                    action['max_vel_x'] = val
                    action['min_vel_x'] = -val
                    action['max_vel_trans'] = val
                else:
                    action['max_vel_x'] = val
        return action

    
    def get_action(self, min, max, value):
        """
        Average value using fixed min and max (so that parameter is in a range).
        """

        return min + (max - min)*((value + 1)/2)
    
    def get_action_for_mdrnn(self, action):
        """
        Concatenate linear velocity, angular velocity, and inflation radius - which is the "action" to send to MDN-RNN Neural Network.
        """

        inflation = action['inflation_radius']
        cmd_vel = self.get_cmd_vel()
        mdrnn_action = np.zeros([1,3])
        mdrnn_action[0,0] = cmd_vel.linear.x
        mdrnn_action[0,1] = cmd_vel.angular.z
        mdrnn_action[0,2] = inflation

        return torch.cuda.FloatTensor(mdrnn_action)

    
    def get_filtered_scan(self):
        """
        Filter scan so that inf values have the maximal value for the lidar scan (3.5).
        """

        scan = self.get_scan()
        ranges = np.asarray(scan.ranges)
        for i in range(0, len(ranges)):
            if ranges[i] == inf:
                ranges[i] = 3.5

        return ranges
    
    def get_vel(self):
        """
        Get current velocity of robot from odometry input.
        """

        odom = self.get_odom()
        vel = np.zeros([2])
        vel[0] = odom.twist.twist.linear.x
        vel[1] = odom.twist.twist.angular.z
        
        return vel
    
    def reward_func(self, obs):
        """
        Reward function for PTDRL. Consists of a combination of robot velocity, and minimal distance to obstacles.
        """

        # Robot velocity
        odom = self.get_odom()
        vel_rob = np.abs(odom.twist.twist.linear.x)

        # Minimum distance
        min_range = np.min(obs)

        reward_func = vel_rob*(-1 if min_range < 0.75 else 1) - self.max_vel_robot # For RL: velocity and obstacles equally

        return reward_func
