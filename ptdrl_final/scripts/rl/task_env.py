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
#dir_path_phydnet = dir_path + "/networks/phydnet.pth"
dir_path_phydnet = dir_path + "/networks/encoder_phydnet_0_10.pth"
dir_path_vae_input = dir_path + "/networks/vae_input_2/best.tar"
dir_path_vae_input_inflation = dir_path + "/networks/vae_input_inflation/best.tar"
dir_path_vae_prediction = dir_path + "/networks/vae_prediction_2/best.tar"
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
        self.timeout_action = 1 # After how many frames to do each action. 3 is around one second. Before 3 # 20 or 10 (2 sec or 1). 6 now 20
        self.num_obs_for_prediction = 3 # How many frames needed for prediction
        self.clear_costmap_counter = 0  
        self.stuck = 0 # if more than 500 seconds passed and the robot did not move then the environment crashed      
        # Position and goals
        self.init_pose = None
        self.goal = None
        #self.offset  = [16.4, 11.9]
        self.offset = [0, 0]
        # Robot constants
        self.max_vel_robot = 1

        self.total_distance_steps = 0
        self.total_distance_no_steps = 0

        # NN's
        self.img_channels = 3
        self.latent_space = 10
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        #self.prediction_module = lstm_mdn_simple.LSTM_MDN(dir_path_checkpoint)

        # Initialize networks
        #self.phydnet = phydnet_predict.PhydNet(dir_path_phydnet)
        #self.vae_input = VAE(self.img_channels, self.latent_space).to(self.device)
        self.vae_input_inflation = VAE(self.img_channels, 32).to(self.device)
        #self.vae_prediction = VAE(self.img_channels, self.latent_space).to(self.device)
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device)
        
        # Load networks
        #self.state_input = torch.load(dir_path_vae_input)
        #self.vae_input.load_state_dict(self.state_input['state_dict'])

        self.state_input_inflation = torch.load(dir_path_vae_input_inflation)
        self.vae_input_inflation.load_state_dict(self.state_input_inflation['state_dict'])

        #state_prediction = torch.load(dir_path_vae_prediction)
        #self.vae_prediction.load_state_dict(state_prediction['state_dict'])

        self.state_mdrnn = torch.load(dir_path_mdrnn)
        self.mdrnn.load_state_dict({k.strip('_l0'): v for k, v in self.state_mdrnn['state_dict'].items()})

        # INitialize netowrk variables
        self.hidden =[torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]

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

        # Local planner
        self.local_planner = data[0]["local_planner"]
        if self.local_planner == "dwa":
            tune_params = "tune_params_dwa"
        else:
            tune_params = "tune_params_teb"

        for i in range(len(data[0][tune_params])):

            self.dict_tune_params[data[0][tune_params][i]["name"]] = [data[0][tune_params][i]["min"], data[0][tune_params][i]["max"]]

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

        return np.zeros([290]) # VAE: 34, VAE_MDRNN: 290, VAE_PHYDNET: 
    
    def _init_env_variables(self):

        self.status_move_base = 0
    
    def step_old(self, action):
        # Set action,  wait for rewards, obs and is_done
        # This function waits for around 1 second to return results.

        # Reward explanation
        # We want to give frequent rewards to encourage:
        # 1) Speed
        # 2) Collision avoidance
        # The way we achieve this is to give a negative reward for each time step, 
        # and a positive reward for distance covered after each time step.
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        #print(self.get_min_scan_buffer_size())
        #self.clear_costmap()
        self.animate_distance_to_obstacle()
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

        total_distance_covered = 0
        start_pos = 0
        final_pos = 0

        for i in range(self.timeout_action):
            self.rate_task.sleep()
            #print(i)
            bad_update = False

            robot_new_position = self.get_odom()
            distance_covered = self.get_covered_distance(robot_position, robot_new_position)
            total_distance_covered += distance_covered
            #print(f"Distance covered is {distance_covered}")
            robot_position = robot_new_position

            is_done = 0
            reward = 0

            ########## To account for bad updates #########
            if distance_covered == 0:
                bad_update = True

            if not bad_update:  ### Distance reward. Reward longer distances.
                reward = self.adjust_covered_distance(distance_covered) # Before -0.01 -0.005. Now it is between 0 and -1
            reward = -0.07
            #print(f"Distance covered: {distance_covered}")
            #print(f"Adjusted distance covered is {reward}")
            hitted_obstacle, dist_to_obstacle = self.hit_obstacle_2()
            if hitted_obstacle or self.status_move_base == 4: # 4 stands for aborted
                #is_done = 1
                #reward = self.reward_dist_to_goal() # Before -1
                if hitted_obstacle:
                    #reward = self.hit_obstacle_reward(dist_to_obstacle) # -30, -40

                    if not bad_update: ### Hitting reward. Punish shorter hits. Between 0 to -3
                        reward += self.adjust_hit_rewards(dist_to_obstacle)
                        #print(f"Hitted reward: {self.adjust_hit_rewards(dist_to_obstacle)}")
                else:
                    reward = -20
                pass
            if self.status_move_base == 4:
                is_done = 1
                reward = -40
            if self.status_move_base == 3: # 3 stands for goal reached
                is_done = 1
                reward = 0
            rewards.append(reward)

            is_dones.append(is_done)

        #print(f"{total_distance_covered} --  {self.get_covered_distance(start_pos, final_pos)}")
            
        obs = self.get_costmap_buffer()

        reward = self.proccess_rewards_3(rewards)
        #if total_distance_covered == 0: # If Gazebo is stuck the robot does not move
            #reward = 0

        mu_i_o, logvar_i_o, mu_p_o, logvar_p_o = self.process_obs_2(obs)
        is_done = self.proccess_is_dones(is_dones)

        obs = self.extract_2(mu_i_o, logvar_i_o, mu_p_o, logvar_p_o)

        #print("stop counting!!!")
        return obs, reward, is_done

    def step_vae(self, action):
        # Step for vae
        # Set action,  wait for rewards, obs and is_done
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
        
        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        mu_i_o, logvar_i_o = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(mu_i_o, logvar_i_o, vel) # Mistake, do not return logvar

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        return obs, reward, is_done

    def step_vae_mdrnn(self, action):
        # Step for vae and mdrnn
        # Update hidden given action, set action,  wait for rewards, return obs and is_done

        # clear buffers
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
        
        # Update hidden
        obs_costmap = self.get_costmap()
        latent_mu, logvar_i_o = self.process_obs_3(obs_costmap)
        mdrnn_action = self.get_action_for_mdrnn(action)
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdrnn(mdrnn_action, latent_mu, self.hidden)


        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        obs_costmap = self.get_costmap()
        latent_mu, _ = self.process_obs_3(obs_costmap)
        vel = self.get_vel()
        obs = self.extract_3(latent_mu, self.hidden[0], vel)

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

        return obs, reward, is_done

    def step_vae_phydnet(self, action):
        # Step for vae and phydnet
        # Set action,  wait for rewards, obs and is_done
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
        
        # Take action
        self.tune_parameters(action)
        self.rate_task.sleep()

        # Get observations
        lidar = self.get_filtered_scan()
        costmap_buffer = self.get_costmap_buffer()
        latent_sequence = self.get_latent_phydnet(costmap_buffer)
        vel = self.get_vel()
        obs = self.extract_4(latent_sequence, vel)

        # Set reward
        reward = self.reward_func(lidar)

        # Episode finished
        if self.status_move_base == 4: # 4 stands for goal aborted
            is_done = 1
            reward = -10
        if self.status_move_base == 3: # 3 stands for goal reached
            is_done = 1
            reward = 0

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
    
    def extract_3(self, mu_i_o, logvar_i_o, vel):
        #mu_i_o = mu_i_o.cpu().data.numpy()[0]
        #logvar_i_o = logvar_i_o.cpu().data.numpy()[0]
        # with torch.no_grad():
        #     vel = torch.cuda.FloatTensor(vel)
        #     mu_i = torch.squeeze(mu_i_o)
        #     logvar_i = torch.squeeze(logvar_i_o)
        #     obs_cat = torch.cat((mu_i, logvar_i, vel), 0)
        #     #obs_cat = obs_cat.cpu().data.numpy()
        #     print(type(obs_cat))

        #     return (obs_cat)
        mu_i_o = mu_i_o.cpu().data.numpy()[0]
        logvar_i_o = logvar_i_o.cpu().data.numpy()[0]

        obs_cat = np.concatenate((mu_i_o, logvar_i_o, vel), axis = 0)

        return (obs_cat)
    
    def extract_4(self, list_tensors, vel):
        obs_cat = np.zeros([0])
        for i in range(len(list_tensors)):
            np_ten = list_tensors[i].cpu().data.numpy()
            obs_cat = np.concatenate((obs_cat, np_ten), axis = 0)

        obs_cat = np.concatenate((obs_cat, vel), axis = 0)

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
        if rewards[0] <= -50: # == -40
            #print("Hitted obstacle!!!")
            print("here")
            return rewards[0]
        return sum(rewards)

    def proccess_rewards_3(self, rewards):
        # Sum all rewards
        return sum(rewards)

    def proccess_is_dones(self, is_dones):
        is_dones.sort()
        if is_dones[-1] == 1:
            return True
        return False
    
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
    
    def phydnet_get_encoder_hidden(self, costmap_sequence):
        # Get data of costmap buffer. Return the encoder_hidden of phydnet.
        video = np.zeros([1,10,1,60,60])
        for itr in range(self.n_frames):
            video[0,itr,0,:,:] = self.costmap_to_np(costmap_sequence[itr])
        encoder_hidden = self.phydnet.get_encoding(video)

        return encoder_hidden
    
    def get_latent_phydnet(self, costmap_buffer):
        # Get data of costmap buffer. Return the VAE encoding of the last 5 frames, and the VAE encoding
        # of the phydnet prediction of the next 5 frames. Latent of phydnet is too big.
        images = np.zeros([10,3,64,64])
        video = np.zeros([1,10,1,60,60])
        for itr in range(self.n_frames):
            video[0,itr,0,:,:] = self.costmap_to_np(costmap_buffer[itr])

        predictions = self.phydnet.predict(video)

        # Concatenate the last 5 frames of costmap_buffer
        for i_frame in range(0,5):
            for i_channel in range(self.img_channels):
                images[i_frame,i_channel,0:60,0:60] = self.costmap_to_np(costmap_buffer[5 + i_frame])
        # Concatenate the predicted next 5 frames
        for i_frame in range(5,10):
            for i_channel in range(self.img_channels):
                images[i_frame,i_channel,0:60,0:60] = np.squeeze(predictions[:,i_frame - 5,:,:])
        
        # Encode everything using VAE
        _, latents, _ = self.vae_input_inflation(torch.cuda.FloatTensor(images))
        latents = latents.reshape(-1)

        #####################################################
        # Show videos #######################################
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Videos')

        # #print("Input video")
        # frames1 = [] # for storing the generated images
        # for i in range(5):
        #     frames1.append([ax1.imshow(np.fliplr(np.squeeze(video[0,i+5,0,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax1.set_title("Input")

        # #print("Prediction video")
        # frames2 = [] # for storing the generated images
        # for i in range(5):
        #     frames2.append([ax2.imshow(np.fliplr(np.squeeze(predictions[0,i,:,:])), cmap=cm.Greys_r,animated=True)])
        # ax2.set_title("Predicted")

        # ani = animation.ArtistAnimation(fig, frames1, interval=500, blit=True)
        # ani2 = animation.ArtistAnimation(fig, frames2, interval=500, blit=True)
        # plt.show(block=False)
        # plt.pause(5)
        # plt.close()
        # plt.show()
        
        return [latents]


    
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
        
    
    def process_obs_3(self, obs):
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

    def process_obs_4(self, obs):

        num_costmaps = 10
        vae_cat = np.zeros([0])
        for itr in range(num_costmaps):
            # Encode costmap/s.
            image_i = np.zeros([1,3,64,64]) # 3 channels. 64x64 costmap (instead of 60x60)

            for itr in range(self.img_channels):
                image_i[0,itr,0:60,0:60] = self.costmap_to_np(obs[itr])
            
            # Encode input image
            image_i = torch.tensor(image_i)
            image_i = image_i.to(self.device, dtype=torch.float)
            image_i_o, mu_i_o, logvar_i_o = self.vae_input_inflation(image_i)

            vae_cat = np.concatenate(vae_cat, mu_i_o, logvar_i_o)


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
        
        return vae_cat
    

    
    def hit_obstacle(self):
        # Get minimum reading of scanner

        scann = self.get_scan()

        if min(scann.ranges) < self.min_dist_to_obstacle:
            #print("Min scan!!!")
            #print(min(scann.ranges))
            return True
        return False

    def hit_obstacle_2(self):
        # Get k minimum readings of scanner
        # k = 5 reafactor 10
        k = 5 # 5 real factor 40

        scann = self.get_scan()
        ranges = np.array(scann.ranges)
        idx = np.argpartition(ranges, k)
        ranges = ranges[idx[:k]]

        if np.max(ranges) < self.min_dist_to_obstacle:
            #print("Min scan!!!")
            #print(ranges)
            return True, np.max(ranges)
        return False, None

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

        return (np.tanh(dist/0.31) - 1) #0.08 without multiplication
    
    def translate_continous_action(self, values):
        # Translate from numbers to values of parameters for parameter tuning

        action = {}
        itr = 0

        for key in self.dict_tune_params:
            if key != 'vel':
                action[key] = self.get_action(self.dict_tune_params[key][0], self.dict_tune_params[key][1], values[itr])
            else:
                if self.local_planner == "dwa":
                    val = self.get_action(self.dict_tune_params[key][0], self.dict_tune_params[key][1], values[itr])
                    action['max_vel_x'] = val
                    action['min_vel_x'] = -val
                    action['max_vel_trans'] = val
                else:
                    val = self.get_action(self.dict_tune_params[key][0], self.dict_tune_params[key][1], values[itr])
                    action['max_vel_x'] = val
            itr += 1
        return action

    
    def get_action(self, min, max, value):

        return min + (max - min)*((value + 1)/2)
    
    def hit_obstacle_reward(self, dist_to_obstacle):

        return -60*(0.4 - dist_to_obstacle)
        
    def adjust_hit_rewards(self, dist_to_obstacle):
        # Reward for proximity to obstacles.
        # Negative reward for distance covered. The more distance covered the smallest the reward. Between 0 and -1
        # Distance covered is usually between 0.2 and 0.02. Typically 0.09

        return 3*(np.tanh(dist_to_obstacle/0.25) - 1) # - Before 8
    
    def get_action_for_mdrnn(self, action):
        inflation = action['inflation_radius']
        cmd_vel = self.get_cmd_vel()
        mdrnn_action = np.zeros([1,3])
        mdrnn_action[0,0] = cmd_vel.linear.x
        mdrnn_action[0,1] = cmd_vel.angular.z
        mdrnn_action[0,2] = inflation

        return torch.cuda.FloatTensor(mdrnn_action)

    
    def animate_distance_to_obstacle(self):
        min_scan_buffer = self.get_min_scan_buffer()
    
    def get_filtered_scan(self):

        scan = self.get_scan()
        ranges = np.asarray(scan.ranges)
        for i in range(0, len(ranges)):
            if ranges[i] == inf:
                ranges[i] = 3.5

        return ranges
    
    def get_vel(self):
        odom = self.get_odom()
        vel = np.zeros([2])
        vel[0] = odom.twist.twist.linear.x
        vel[1] = odom.twist.twist.angular.z
        
        return vel
    
    def is_robot_stuck(self):
        # If the robot absoulte velocities sum to a number smaller than 1 in the last 100 steps (10 seconds)
        # the robot is stuck
        odom  = self.get_odom_buffer()
        s = sum(odom)
        info = None
        if s < 1:
            self.stuck += 1
        else:
            self.stuck = 0

        if self.stuck >= 50:
            info = "Environment crashed"

        return 
    
    def reward_func(self, obs):
        # Robot velocity
        odom = self.get_odom()
        vel_rob = np.abs(odom.twist.twist.linear.x)

        # Minimum distance
        min_range = np.min(obs)

        #print(vel_rob)
        #print(min_range)

        ###########################################################################################################
        ##### Reward function Bayesian - Immediate reward: velocity of robot + distance + relative velocity #######
        ###########################################################################################################

        #reward_func = vel_rob*(-1 if min_range < 0.75 else 1) - self.max_vel_robot # For RL: velocity and obstacles equally
        reward_func = (np.tanh((vel_rob-1)*4)+1)*(-50 if min_range < 0.75 else 0) - 0.2 # For RL: penalize greatly high velocities near obstacles
        #print(f"vel_rob: {vel_rob}, reward_func: {reward_func}")
        #reward_func = vel_rob*(-1 if min_range < 0.75 else 1)

        #reward_func = 4*vel_rob*(-1 if min_range < 0.8 else 1) - 0.1*(-1 if min_range < 0.5 else 0)
        #reward_func = 4*vel_rob*np.tanh((min_range-0.4)/0.2) - 0.1*(1/min_range)
        #reward_func = 4*vel_rob*np.tanh((min_range-0.8))
        #print(reward_func)
        #reward_func = - 0.1*(1/min_range)
        #print(reward_func)

        ###########################################################################################################
        #### Reward function DRL - Accumulated reward: velocity of robot when close to obstacles + distance - 1 ###
        ###########################################################################################################

        #reward_func = vel_rob*np.tanh((min_range-0.8)*4 if min_range < 0.8 else 0) - 0.1*(1/min_range) - 1
        #reward_func = vel_rob*np.tanh((min_range-0.8)*4 if min_range < 0.8 else 0) 
        #reward_func = - 0.1*(1/min_range)
        #reward_func = - 1

        return reward_func
    











