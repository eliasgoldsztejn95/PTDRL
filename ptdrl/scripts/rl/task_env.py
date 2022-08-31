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

import matplotlib.cm as cm
from random import randrange
import matplotlib.animation as animation
from PIL import Image

import matplotlib.pyplot as plt

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))

dir_path_phydnet = dir_path + "/networks/phydnet.pth"
dir_path_vae_input = dir_path + "/networks/vae_input_2/best.tar"
dir_path_vae_prediction = dir_path + "/networks/vae_prediction_2/best.tar"

dir_path_yaml = dir_path
dir_path_yaml = dir_path_yaml.split("/")
dir_path_yaml = dir_path_yaml[:-2]
dir_path_yaml += ["params"]
dir_path_yaml = '/'.join(dir_path_yaml)

file = "task_params"
yaml_file = "/" + file + ".yaml"

dir_path_yaml += yaml_file

class PtdrlTaskEnv(robot_env.PtdrlRobEnv):

    """ Superclass for PTDRL Task environment
    """
    def __init__(self):
        """Initializes a new PTDRL Task environment
        """

        self.get_params()

        self.rate_task = rospy.Rate(3)

        self.counter = -1 #1 Counter to change goal and init pose
        self.timeout_action = 6 # After how many frames to do each action. 3 is around one second. Before 3
        self.num_obs_for_prediction = 3 # How many frames needed for prediction
        
        self.init_pose = None
        self.goal = None
        #self.offset  = [16.4, 11.9]
        self.offset = [0, 0]

        # NN's
        self.img_channels = 3
        self.latent_space = 10
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        #self.prediction_module = lstm_mdn_simple.LSTM_MDN(dir_path_checkpoint)

        self.phydnet = phydnet_predict.PhydNet(dir_path_phydnet)

        self.vae_input = VAE(self.img_channels, self.latent_space).to(self.device)
        self.vae_prediction = VAE(self.img_channels, self.latent_space).to(self.device)

        self.state_input = torch.load(dir_path_vae_input)
        self.vae_input.load_state_dict(self.state_input['state_dict'])

        state_prediction = torch.load(dir_path_vae_prediction)
        self.vae_prediction.load_state_dict(state_prediction['state_dict'])

        robot_env.PtdrlRobEnv.__init__(self, model_name = self.model_name, amcl = self.amcl, min_dist_to_obstacle = self.min_dist_to_obstacle,
                                    min_dist_to_goal = self.min_dist_to_goal, num_tracks = self.num_tracks, timeout = self.timeout)

    def get_params(self):

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
        self.dict_tune_params = {}

        # World
        self.min_dist_to_obstacle = data[0]["min_dist_to_obstacle"]
        self.min_dist_to_goal = data[0]["min_dist_to_goal"]

        self.list_init_pose = []
        self.list_goals = []

        for i in range(len(data[0]["tune_params"])):

            self.dict_tune_params[data[0]["tune_params"][i]["name"]] = [data[0]["tune_params"][i]["min"], data[0]["tune_params"][i]["max"]]

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

        self.init_robot(self.init_pose)
        #print(self.init_pose)
        #print("Robot initialized!!!")
    
    def _send_goal(self):

        #print(self.goal)
        self.send_goal(self.goal)
        #print("Goal sent!!!")
    
    def reset(self):
        # Init variables, set robot in position, clear costmap and send goal.

        self.counter += 1
        iter = self.counter % len(self.list_init_pose)

        self.init_pose = self.list_init_pose[iter]
        self.goal = self.list_goals[iter]

        self._init_env_variables()

        self._set_init_pose()

        self.clear_costmap()

        self._send_goal()

        return np.zeros([40])
    
    def _init_env_variables(self):

        self.status_move_base = 0
    
    def step(self, action):
        # Set action,  wait for rewards, obs and is_done
        # This function waits for around 1 second to return results.

        # Reward explanation
        # We want to give frequent rewards to encourage:
        # 1) Speed
        # 2) Collision avoidance
        # The way we achieve this is to give a negative reward for each time step, 
        # and a positive reward for distance covered after each time step.
        distance_covered = 0
        robot_position = self.get_odom()

        rewards = []
        obs = []
        is_dones = []
        #print("start counting !!!")
        action = action
        if self.discrete_actions == False:
            action = self.translate_continous_action(action)
        
        self.tune_parameters(action)

        for i in range(self.timeout_action):
            self.rate_task.sleep()

            distance_covered = self.get_covered_distance(robot_position, self.get_odom())
            #print(f"Distance covered is {distance_covered}")
            robot_position = self.get_odom()

            is_done = 0
            reward = self.adjust_covered_distance(distance_covered) # Before -0.01 -0.005. Now it is between 0 and -1
            #print(f"Adjusted distance covered is {reward}")
            if self.hit_obstacle_2() or self.status_move_base == 4: # 4 stands for aborted
                #is_done = 1
                #reward = self.reward_dist_to_goal() # Before -1
                reward = -40 # -30
            if self.status_move_base == 4:
                is_done = 1
            if self.status_move_base == 3: # 3 stands for goal reached
                is_done = 1
                reward = 0
            rewards.append(reward)

            is_dones.append(is_done)
            
        obs = self.get_costmap_buffer()

        reward = self.proccess_rewards_2(rewards)
        mu_i_o, logvar_i_o, mu_p_o, logvar_p_o = self.process_obs_2(obs)
        is_done = self.proccess_is_dones(is_dones)

        obs = self.extract_2(mu_i_o, logvar_i_o, mu_p_o, logvar_p_o)

        if is_done == 1:
            scann = self.get_scan()
        #print("stop counting!!!")
        return obs, reward, is_done
    
    def extract(self, obs, prediction):

        obs = obs[:,-1,:]
        prediction_mu = prediction[0]
        prediction_sigma = prediction[1]

        prediction_mu = prediction_mu.detach()
        prediction_mu = prediction_mu.numpy()
        prediction_mu = prediction_mu[:,-1,:]

        prediction_sigma = prediction_sigma.detach()
        prediction_sigma = prediction_sigma.numpy()
        prediction_sigma = prediction_sigma[:,-1,:]

        obs_cat = np.concatenate((obs[0], prediction_mu[0], prediction_sigma[0]), axis = 0)

        return (obs_cat)
    
    def extract_2(self, mu_i_o, logvar_i_o, mu_p_o, logvar_p_o):
        mu_i_o = mu_i_o.cpu().data.numpy()[0]
        logvar_i_o = logvar_i_o.cpu().data.numpy()[0]
        mu_p_o = mu_p_o.cpu().data.numpy()[0]
        logvar_p_o = logvar_p_o.cpu().data.numpy()[0]

        obs_cat = np.concatenate((mu_i_o, logvar_i_o, mu_p_o, logvar_p_o), axis = 0)

        return (obs_cat)


    def proccess_rewards(self, rewards):
        # If hitted return hitted, else return reached, else retun time penalty.

        rewards.sort()
        if rewards[0] != -0.005:
            return rewards[0]
        return rewards[-1]

    def proccess_rewards_2(self, rewards):
        # If hitted return hitted, else return distance covered.
        
        rewards.sort()
        if rewards[0] == -40:
            print("Hitted obstacle!!!")
            return rewards[0]
        return sum(rewards)

    def proccess_is_dones(self, is_dones):
        is_dones.sort()
        if is_dones[-1] == 1:
            return 1
        return 0
    
    def process_obs(self, obs):
        # Get x and y pos of each marker, append the last n frames. Predcit and return last
        # input frame, and the last predicted frame.
        np_obs = np.zeros([1,self.num_obs_for_prediction,(self.num_tracks)*2])

        if len(obs) >= self.num_obs_for_prediction:
            num_obs = self.num_obs_for_prediction
        else:
            num_obs = len(obs)

        for iter in range(len(obs) - num_obs, len(obs),1):
            i = iter - len(obs) + num_obs
            for track in range(self.num_tracks):
                np_obs[:, i, track*2] = obs[i].markers[track].pose.position.x
                np_obs[:, i, track*2 + 1] = obs[i].markers[track].pose.position.y
        
        for iter in range(self.num_obs_for_prediction - num_obs):
            i = iter + num_obs
            for track in range(self.num_tracks):
                np_obs[:, i, track*2] = obs[-1].markers[track].pose.position.x
                np_obs[:, i, track*2 + 1] = obs[-1].markers[track].pose.position.y
        
        prediction = self.prediction_module.predict(np_obs)

        return np_obs, prediction

    def costmap_to_np(self, costmap):
        # Occupancy grid to numpy
        costmap_np = np.zeros([60,60])
        if costmap is not None:
            costmap_np = np.divide(np.array_split(costmap.data, 60),100) # costmap size is 60
        return costmap_np
    
    def process_obs_2(self, obs):
        # Get data of costmap buffer. Return the encoding of the last frame (the current costmap).
        # predict the next n frames. Return the encoding of the last frame.
        image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)
        video = np.zeros([1,10,1,60,60])
        image_p = np.zeros([1,3,64,64])

        for itr in range(self.img_channels):
            image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs[-1])
          
        for itr in range(self.n_frames):
            video[0,itr,0,:,:] = self.costmap_to_np(obs[itr])
        
        predictions = self.phydnet.predict(video)
        for itr in range(self.img_channels):
            image_p[0,itr,0:60,0:60] = np.squeeze(predictions[:,-1,:,:])

        # Encode input image
        image_i = torch.tensor(image_i)
        image_i = image_i.to(self.device, dtype=torch.float)
        image_i_o, mu_i_o, logvar_i_o = self.vae_input(image_i)

        # Encode predicted image
        image_p = torch.tensor(image_p)
        image_p = image_p.to(self.device, dtype=torch.float)
        image_p_o, mu_p_o, logvar_p_o  = self.vae_prediction(image_p)

        #####################################################
        # Show videos #######################################
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Videos')

        # #print("Input video")
        # frames1 = [] # for storing the generated images
        # for i in range(10):
        #     frames1.append([ax1.imshow(np.fliplr(np.squeeze(video[0,i,0,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax1.set_title("Input")

        # #print("Prediction video")
        # frames2 = [] # for storing the generated images
        # for i in range(10):
        #     frames2.append([ax2.imshow(np.fliplr(np.squeeze(predictions[0,i,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax2.set_title("Predicted")

        # ani = animation.ArtistAnimation(fig, frames1, interval=100, blit=True)
        # ani2 = animation.ArtistAnimation(fig, frames2, interval=100, blit=True)

        ####################################################
        ####################################################

        ####################################################
        #Show images #######################################

        # fig2, (ax11, ax22) = plt.subplots(1, 2)
        # fig2.suptitle('Images')

        # im1 = Image.fromarray(np.fliplr(np.squeeze(image_i_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax11.imshow(im1)
        # ax11.set_title("Decoded input")


        # im3 = Image.fromarray(np.fliplr(np.squeeze(image_p_o[0,0].cpu().data.numpy()))*256)
        # imgplot = ax22.imshow(im3)
        # ax22.set_title("Decoded prediction")

        #####################################################
        #####################################################

        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
        #plt.show()
        
        return mu_i_o, logvar_i_o, mu_p_o, logvar_p_o
    
    def hit_obstacle(self):
        # Get minimum reading of scanner

        scann = self.get_scan()

        if min(scann.ranges) < self.min_dist_to_obstacle:
            print("Min scan!!!")
            print(min(scann.ranges))
            return True
        return False

    def hit_obstacle_2(self):
        # Get k minimum readings of scanner
        # k = 5 reafactor 10
        k = 7 # 5 real factor 40

        scann = self.get_scan()
        ranges = np.array(scann.ranges)
        idx = np.argpartition(ranges, k)
        ranges = ranges[idx[:k]]

        if np.max(ranges) < self.min_dist_to_obstacle:
            print("Min scan!!!")
            print(ranges)
            return True
        return False

    def goal_reached(self):
        # Check if robot reached destination

        status_move_base = self.get_status_move_base()

        return status_move_base == 3
    
    def dist_to_goal(self):
        odom = self.get_odom()
        
        return np.sqrt((odom.pose.pose.position.x - self.goal.position.x + self.offset[0])**2 + (odom.pose.pose.position.y - self.goal.position.y + self.offset[1])**2)
    
    def reward_dist_to_goal(self):

        dist = self.dist_to_goal()
        return 2*(np.tanh(2 - dist/5))
    
    def get_covered_distance(self, pos1, pos2):
        # Get euclidean distance between two positions

        return np.sqrt((pos1.pose.pose.position.x - pos2.pose.pose.position.x)**2 + (pos1.pose.pose.position.y - pos2.pose.pose.position.y)**2)
    
    def adjust_covered_distance(self, dist):
        # Negative reward for distance covered. The more distance covered the smallest the reward. Between 0 and -1
        # Distance covered is usually between 0.2 and 0.02. Typically 0.09

        return (np.tanh(dist/0.08) - 1)
    
    def translate_continous_action(self, values):
        # Translate from numbers to values of parameters for parameter tuning

        action = {}
        itr = 0

        for key in self.dict_tune_params:
            if key != 'vel':
                action[key] = self.get_action(self.dict_tune_params[key][0], self.dict_tune_params[key][1], values[itr])
            else:
                val = self.get_action(self.dict_tune_params[key][0], self.dict_tune_params[key][1], values[itr])
                action['max_vel_x'] = val
                action['min_vel_x'] = -val
                action['max_vel_trans'] = val
            itr += 1
        
        return action
    
    def get_action(self, min, max, value):

        return min + (max - min)*((value + 1)/2)
        






