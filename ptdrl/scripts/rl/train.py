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
import task_env
import yaml
from yaml.loader import SafeLoader

def main():
    a = task_env.PtdrlTaskEnv()
    a.start_epsiode()

if __name__ == '__main__':
    rospy.init_node('init_training')
    main()