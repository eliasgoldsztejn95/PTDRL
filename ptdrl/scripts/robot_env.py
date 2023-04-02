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
import copy
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
from nav_msgs.msg import Odometry, OccupancyGrid
from actionlib_msgs.msg import GoalID
import csv
import os
from pathlib import Path
from std_srvs.srv import EmptyRequest,  Empty
from visualization_msgs.msg import MarkerArray, Marker
import dynamic_reconfigure.client

class PtdrlRobEnv():
    """ Superclass for PTDRL Robot environment
    """

    def __init__(self, model_name, amcl, min_dist_to_obstacle, min_dist_to_goal, num_tracks, timeout):
        """Initializes a new PTDRL environment
        """
        # Internal variables

        # Time
        self.rate_rob = rospy.Rate(10)
        self.timeout = timeout

        # Model
        self.model_name = model_name

        # Navigation
        self.amcl = amcl
        self.status_move_base = 0

        # Tracking
        self.num_tracks = num_tracks # NUmber of clusters to track
        self.n_frames = 10 # Number of costmap frames for video prediction
        self.counter = 0 # Counter to 
        self.costmap_bufferr = [None]*10 # Buffer to store the last 10 costmaps. 60x60
        self.k = 1 # Minimum number of distances
        self.min_scan_buffer = [None]*10 # Buffer to store the last 10 min scans.
        self.min_scan_buffer_size = 0 # How many scan readings are done at each step
        self.odom_buffer = [None]*10 # Buffer to store the last 10 velocities.
        self.odom_buffer_size = 0 # How many scan readings are done at each step

        # Sensors
        self.scann = LaserScan()
        self.odomm = Odometry()
        self.vizz = MarkerArray()
        for i in range(self.num_tracks):
            marker = Marker()
            marker.id = i
            self.vizz.markers.append(marker)
        self.costmapp = OccupancyGrid()

        # Local planner
        self.cmd_vell = Twist()

        # World
        self.min_dist_to_obstacle = min_dist_to_obstacle
        self.min_dist_to_goal = min_dist_to_goal
        
        # Start suscribers and publishers

        # Suscribers

        # Sensors
        print("Suscribing to scan")
        rospy.Subscriber("scan", LaserScan, self._scan_callback)
        print("Suscribed!")

        print("Suscribing to odom")
        rospy.Subscriber("odom", Odometry, self._odom_callback)
        print("Suscribed!")

        # Local planner
        print("Suscribing to cmd_vel")
        rospy.Subscriber('/pedbot/control/cmd_vel', Twist, self._cmd_vel_callback)
        print("Suscribed!")

        # Tracking
        # print("Suscribing to viz")
        # rospy.Subscriber("viz", MarkerArray, self._viz_callback)
        # print("Suscribed!")

        # Costmap
        print("Suscribing to costmap")
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self._costmap_callback)
        print("Suscribed!")

        # Publishers
        self._vel_pub = rospy.Publisher('/pedbot/control/cmd_vel', Twist, queue_size=10)

        # Services
        print("Waiting for gazebo server")
        rospy.wait_for_service('/gazebo/set_model_state')
        self._set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        print("Connected!")

        print("Waiting for move_base server")
        rospy.wait_for_service('/move_base/clear_costmaps')
        self._clear_costmap_service = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        print("Connected!")


        # Actions
        self._move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move base action")
        self._move_base_client.wait_for_server()

        self._tuning_client_local = dynamic_reconfigure.client.Client('move_base/DWAPlannerROS',timeout=4, config_callback=None)
        #self._tuning_client_local = dynamic_reconfigure.client.Client('move_base/TebLocalPlannerROS',timeout=4, config_callback=None)
        self._tuning_client_inflation = dynamic_reconfigure.client.Client('move_base/local_costmap/inflation_layer',timeout=4, config_callback=None)

        

                # Launch init function
        #super(PtdrlRobEnv, self).__init__(model_name = self.model_name, amcl = self.amcl, min_dist_to_obstacle = self.min_dist_to_obstacle,
                                       # min_dist_to_goal = self.min_dist_to_goal, num_tracks = self.num_tracks, timeout = self.timeout)

        super(PtdrlRobEnv, self).__init__()

        # RobotEnv methods

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

        self.update_min_scan_buffer_k()
        
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
        self.update_odom_buffer()
    
    def _viz_callback(self, msg):

        # Update viz reading
        for i in range(self.num_tracks):
            self.vizz.markers[i].pose.position.x = msg.markers[i].pose.position.x
            self.vizz.markers[i].pose.position.y = msg.markers[i].pose.position.y

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
        #print("1")

        self.update_buffer()
    
    def _cmd_vel_callback(self, msg):
        self.cmd_vell.linear.x = msg.linear.x
        self.cmd_vell.angular.z = msg.angular.z

    # Methods that the TaskEnvironment will need to define here as virtual

    def _set_init_pose(self):
        """Sets the robot in its init pose.
        """
        raise NotImplementedError()
    
    def _send_goal(self):
        """Send goal via move base.
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initalized at the star of each episode.
        """
        raise NotImplementedError()
    
    def step(self, action):
        """Applies given action. Calculates the reward. Gets observations. Checks if episode is done.
        """
        raise NotImplementedError()

    # Methods that TakEnvironment needs

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
        if inflation:
            self._tuning_client_inflation.update_configuration(inflation_param)


    def init_robot(self, init_pose):
        ##################################################
        ########## Respawn Robot in Gazebo ###############
        ###################################################

        set_model_state_object = SetModelStateRequest()
        # Gazebo - respawn robot
        set_model_state_object.model_state.model_name = self.model_name
        set_model_state_object.model_state.reference_frame = 'world'
        set_model_state_object.model_state.pose.position.x = init_pose.position.x
        set_model_state_object.model_state.pose.position.y = init_pose.position.y
        set_model_state_object.model_state.pose.orientation.x = init_pose.orientation.x
        set_model_state_object.model_state.pose.orientation.y = init_pose.orientation.y
        set_model_state_object.model_state.pose.orientation.z = init_pose.orientation.z
        set_model_state_object.model_state.pose.orientation.w = init_pose.orientation.w
        result = self._set_model_state_service(set_model_state_object)
        self.freeze_robot()
        #print("Setting" + self.model_name + "result: " + str(result))
    
    def freeze_robot(self):
        # Publish vel 0 to robot
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.linear.y = 0
        cmd_vel.angular.z = 0
        for i in range(0,5):
            self._vel_pub.publish(cmd_vel)
            self.rate_rob.sleep()
    
    def send_goal(self, goal):
        #############################################
        ########## Send goal to move base ###########
        #############################################

        # Goal posisition
        goal_pose = Pose()
        goal_pose.position.x = goal.position.x
        goal_pose.position.y = goal.position.y

        # Send simple goal to move_base

        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = 'map' 
        move_base_goal.target_pose.pose.position.x = goal_pose.position.x
        move_base_goal.target_pose.pose.position.y = goal_pose.position.y
        move_base_goal.target_pose.pose.orientation.z = 0.727
        move_base_goal.target_pose.pose.orientation.w = 0.686

        self._move_base_client.send_goal(move_base_goal, done_cb=self._move_base_callback, feedback_cb=None)
    
    def _move_base_callback(self, state, result):

        self.status_move_base = state
        #print("Status move base callback: " + str(state))
    
    def clear_costmap(self):
        # Clear costmaps

        clear_costmap_object = EmptyRequest()

        result = self._clear_costmap_service(clear_costmap_object)
        #print("Clearing costmaps" + str(result))
    
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
    
    def get_viz(self):
        viz = MarkerArray()
        for i in range(self.num_tracks):
            marker = Marker()
            marker.id = i
            viz.markers.append(marker)

        for i in range(self.num_tracks):
            viz.markers[i].pose.position.x = self.vizz.markers[i].pose.position.x
            viz.markers[i].pose.position.y = self.vizz.markers[i].pose.position.y

        return viz
    
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
    
    def update_buffer(self):
        # Updates costmap buffer

        self.costmap_bufferr.pop(0)
        self.costmap_bufferr.append(self.get_costmap())
    
    def update_min_scan_buffer(self):

        scann = self.get_scan()
        min_scan = np.min(scann.ranges)
        self.min_scan_buffer.pop(0)
        self.min_scan_buffer.append(min_scan)
        self.min_scan_buffer_size += 1

    def update_odom_buffer(self):

        odomm = self.get_odom()
        linear_vel = np.abs(odomm.twist.twist.linear.x)
        self.odom_buffer.pop(0)
        self.odom_buffer.append(linear_vel)
        self.odom_buffer_size += 1

    def update_min_scan_buffer_k(self):

        scann = self.get_scan()
        ranges = np.array(scann.ranges)
        idx = np.argpartition(ranges, self.k)
        ranges = ranges[idx[:self.k]]
        min_max_dist = np.max(ranges)

        self.min_scan_buffer.pop(0)
        self.min_scan_buffer.append(min_max_dist)
        self.min_scan_buffer_size += 1

    def get_costmap_buffer(self):

        costmap_buffer = [None]*10
        for itr in range(10):
            costmap_buffer[itr] = self.costmap_bufferr[itr]
        
        return costmap_buffer
    
    def get_min_scan_buffer(self):

        min_scan_buffer = [None]*10
        for itr in range(10):
            min_scan_buffer[itr] = self.min_scan_buffer[itr]
        
        #print(min_scan_buffer)
        return min_scan_buffer

    def get_odom_buffer(self):

        odom_buffer = [None]*10
        for itr in range(10):
            odom_buffer[itr] = self.odom_buffer[itr]
        
        #print(min_scan_buffer)
        return odom_buffer
    
    def get_linear_vel(self):
        odomm = self.get_odom()
        linear_x = np.abs(odomm.twist.twist.linear.x)

        return linear_x

    def get_status_move_base(self):
        status_move_base = self.status_move_base
        return status_move_base

    def get_min_scan_buffer_size(self):
        min_scan_buffer_size = self.min_scan_buffer_size
        return min_scan_buffer_size

    def get_odom_buffer_size(self):
        odom_buffer_size = self.odom_buffer_size
        return odom_buffer_size
    
    def clear_min_scan_buffer_size(self):
        self.min_scan_buffer_size = 0

    def clear_odom_buffer_size(self):
        self.odom_buffer_size = 0


