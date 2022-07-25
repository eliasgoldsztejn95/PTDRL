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

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))

dir_path_checkpoint = dir_path
dir_path_checkpoint += "/checkpoint.pt"

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

        self.counter = -1
        self.timeout_action = 3
        self.num_obs_for_prediction = 3
        
        self.init_pose = None
        self.goal = None
        self.offset  = [16.4, 11.9]

        self.prediction_module = lstm_mdn_simple.LSTM_MDN(dir_path_checkpoint)

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

        # World
        self.min_dist_to_obstacle = data[0]["min_dist_to_obstacle"]
        self.min_dist_to_goal = data[0]["min_dist_to_goal"]

        self.list_init_pose = []
        self.list_goals = []

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
        print(self.init_pose)
        print("Robot initialized!!!")
    
    def _send_goal(self):

        print(self.goal)
        self.send_goal(self.goal)
        print("Goal sent!!!")
    
    def start_epsiode(self):
        # Init variables, set robot in position, clear costmap and send goal.

        self.counter += 1
        iter = self.counter % len(self.list_init_pose)

        self.init_pose = self.list_init_pose[iter]
        self.goal = self.list_goals[iter]

        self._init_env_variables()

        self._set_init_pose()

        self.clear_costmap()

        self._send_goal()
    
    def _init_env_variables(self):

        self.status_move_base = 0
    
    def _set_action_get_reward_obs_is_done(self, action):
        # Set action,  wait for rewards, obs and is_done
        # This function waits for 1 second to return results.

        rewards = []
        obs = []
        is_dones = []

        self.tune_parameters(action)

        for i in range(self.timeout_action):
            self.rate_task.sleep()
            reward = 0
            if self.hit_obstacle() or self.status_move_base == 4:
                reward = -1
            if self.status_move_base == 3:
                reward = 1
            rewards.append(reward)

            is_done = 0
            if reward != 0:
                is_done = 1
            is_dones.append(is_done)
            
            obs.append(self.get_obs())

        reward = self.proccess_rewards(rewards)
        obs, prediction = self.process_obs(obs)
        is_done = self.proccess_is_dones(is_dones)

        return (obs, prediction), reward, is_done

    def proccess_rewards(self, rewards):
        
        rewards.sort()
        if rewards[0] == -1:
            return -1
        return rewards[-1]

    def proccess_is_dones(self, is_dones):
        is_dones.sort()
        if is_dones[-1] == 1:
            return 1
        return 0
    
    def process_obs(self, obs):
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

        return np_obs[:,:,:], prediction
    
    def hit_obstacle(self):
        # Get minimum reading of scanner

        scann = self.get_scan()

        if min(scann.ranges) < self.min_dist_to_obstacle:
            return True
        return False

    def goal_reached(self):
        # Check if robot reached destination

        status_move_base = self.get_status_move_base()

        return status_move_base == 3






