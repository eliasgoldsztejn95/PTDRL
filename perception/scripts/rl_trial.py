#!/usr/bin/env python3

###########################################
## Start test for navigation performance ##
###########################################

# When called, this service receives the starting position
# of the robot and its goal. It spawns the robot where
# specified, updates the pose and sends the goal to 
# move_base.


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
import random


# Global variables

# Model
model_name = "turtlebot3_waffle"

# Navigation
covariance = np.zeros([36])
amcl = False
move_base = True
status_move_base = 0
offset = [16.4, 11.9]

# Tracking
num_tracks = 6

# Sensors
scann = LaserScan()
odomm = Odometry()
vizz = MarkerArray()

for i in range(num_tracks):
    marker = Marker()
    marker.id = i
    vizz.markers.append(marker)

# Tuning
def clb_config(msg):
    print(msg)
    print("Updated parameters")
tuning_client = dynamic_reconfigure.client.Client('move_base/DWAPlannerROS',timeout=4, config_callback=clb_config)

# World
min_dist_to_obstacle = 0.15
min_dist_to_goal = 0.2
goal = Pose()

# Test
dir_path = os.path.dirname(os.path.realpath(__file__))
test_file = dir_path + "/score.csv"
repetitions = 2000
timeout = 700 # 70 seconds

def one_covariance():
    global covariance
    for i in range (0,5):
        for j in range (0,5):
            if i==j:
                covariance[i*6 + j] = 1.0
    covariance[-1] = 1.0
#one_covariance()
#print(covariance)


def init_robot(init_pose):
    ##########################################################################
    ########## Respawn Robot in Gazebo - update AMCL if needed ###############
    ##########################################################################
    global covariance

    # Start being a client of /gazebo/set_model_state
    rospy.wait_for_service('/gazebo/set_model_state')
    set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_object = SetModelStateRequest()

    # Gazebo - respawn robot
    set_model_state_object.model_state.model_name = model_name
    set_model_state_object.model_state.reference_frame = 'world'
    set_model_state_object.model_state.pose.position.x = init_pose.position.x
    set_model_state_object.model_state.pose.position.y = init_pose.position.y
    set_model_state_object.model_state.pose.orientation.x = init_pose.orientation.x
    set_model_state_object.model_state.pose.orientation.y = init_pose.orientation.y
    set_model_state_object.model_state.pose.orientation.z = init_pose.orientation.z
    set_model_state_object.model_state.pose.orientation.w = init_pose.orientation.w
    result = set_model_state_service(set_model_state_object)
    freeze_robot()
    print("Setting" + model_name + "result: " + str(result))

    # If using AMCL send inital position of robot
    if amcl:
        pub_pose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        amcl_pose = PoseWithCovarianceStamped()
        amcl_orientation = quaternion_from_euler(offset[2],0,0)
        amcl_pose.header.frame_id = 'world'
        amcl_pose.pose.pose.position.x = init_pose.position.x + offset[0]
        amcl_pose.pose.pose.position.y = init_pose.position.y + offset[1]
        #amcl_pose.pose.pose.position.z = init_pose.position.z
        #amcl_pose.pose.pose.orientation.x = init_pose.orientation.x + offset[0]
        #amcl_pose.pose.pose.orientation.y = init_pose.orientation.y + offset[1]
        #amcl_pose.pose.pose.orientation.z = init_pose.orientation.z 
        #amcl_pose.pose.pose.orientation.w = init_pose.orientation.w
        amcl_pose.pose.covariance = covariance
        pub_pose.publish(amcl_pose)
        print("AMCL pose published")
    

def freeze_robot():
    # Publish vel 0 to robot
    rate = rospy.Rate(10)
    pub_vel = rospy.Publisher('/pedbot/control/cmd_vel', Twist, queue_size=10)
    cmd_vel = Twist()
    cmd_vel.linear.x = 0
    cmd_vel.linear.y = 0
    cmd_vel.angular.z = 0
    for i in range(0,5):
        pub_vel.publish(cmd_vel)
        rate.sleep()

def send_goal():
    ##########################################################################
    ########## Publish goal position - send to move_base if needed ###########
    ##########################################################################
    global goal

    # Goal posisition
    goal_pose = Pose()
    goal_pose.position.x = goal.position.x
    goal_pose.position.y = goal.position.y

    if move_base:
        # Send simple goal to move_base

        client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move base server")
        client.wait_for_server()
        print("Goal received")

        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = 'map' 
        move_base_goal.target_pose.pose.position.x = goal_pose.position.x
        move_base_goal.target_pose.pose.position.y = goal_pose.position.y
        move_base_goal.target_pose.pose.orientation.z = 0.727
        move_base_goal.target_pose.pose.orientation.w = 0.686

        client.send_goal(move_base_goal, done_cb=clb_move_base_done, feedback_cb=clb_move_base_feedback)
        return client
    else:
        # Publish goal to topic for robots without move_base
        simple_goal = Pose()
        simple_goal.position.x = goal_pose.position.x
        simple_goal.position.y = goal_pose.position.y
        pub_goal = rospy.Publisher('/goal_pose', Pose, queue_size=10)
        publish_once(pub_goal, simple_goal)

        return None

def init_test():
    # Run several tests
    global num_tests

    for i in range(0,repetitions):
        run_test()


def run_test():
    # Set robot and send goal
    global goal
    global scann
    global status_move_base
    global tuning_client

    print("#### Initializing test ####\n")
    #### Set move_base status ####
    status_move_base = 0

    #### Initialize robot ###
    init_pose = Pose()
    init_pose.position.x = 5
    init_pose.position.y = 5
    init_pose.position.z = 0
    orientation = quaternion_from_euler(0,0,0)
    
    init_pose.orientation.x =  orientation[0]
    init_pose.orientation.y =  orientation[1]
    init_pose.orientation.z =  orientation[2]
    init_pose.orientation.w =  orientation[3]
    
    init_robot(init_pose)

    ### Clear costmap ###
    print("clearing costmap")
    clear_costmap()

    ### Tune parameters
    max_vel = random.uniform(4.0, 5.0)
    params = { 'max_vel_x' : max_vel}
    config = tuning_client.update_configuration(params)
    print(f"Tuning DWA parameters {config['max_vel_x']}")

    ### Send goal to robot ###
    goal.position.x = 25
    goal.position.y = -5
    client = send_goal()

    #### Wait for robot to end test ###
    test_robot()
    print("#### Test terminated ####\n")


def clear_costmap():
    # Clear costmaps

    # Start being a client of /move_base/clear_costmaps
    rospy.wait_for_service('/move_base/clear_costmaps')
    clear_costmap_service = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
    clear_costmap_object = EmptyRequest()

    result = clear_costmap_service(clear_costmap_object)
    print("Clearing costmaps" + str(result))

def clb_scan(msg):
    # Update lidar reading
    global scann

    current_time = rospy.Time.now()
    scann.header.stamp = current_time
    scann.header.frame_id = msg.header.frame_id 
    scann.angle_min = msg.angle_min
    scann.angle_max = msg.angle_max
    scann.angle_increment = msg.angle_increment
    scann.time_increment = msg.time_increment
    scann.range_min = msg.range_min
    scann.range_max = msg.range_max
    scann.ranges = msg.ranges
    scann.intensities = msg.intensities



def test_robot():
    # Wait for completion of one test. The robot reached the goal, hitted an obstacle or move_base was aborted
    global move_base

    time_counter = 0
    rate = rospy.Rate(10)
    if move_base:
        while not rospy.is_shutdown():
            if hit_obstacle() or status_move_base == 3 or status_move_base == 4 or time_counter==timeout:
                break
            time_counter += 1
            rate.sleep() 
    else:
        while not rospy.is_shutdown():
            if hit_obstacle() or goal_reached() or time_counter==timeout:
                break
            time_counter += 1
            rate.sleep() 
    
    test_termination(time_counter)

def test_termination(time_counter):
    # Save termination results for one test
    global odomm

    if hit_obstacle():
        print("Obstacle hitted!!!!")
        print(f"Distance to object is {min(scann.ranges)}, Distance to goal is: {dist_to_goal()}")
    
    if goal_reached():
        print("Reached goal!!!!")
        print(f"Distance to goal is: {dist_to_goal()}")
    
    if move_base:
        if status_move_base == 4:
            print("Move base failed to produce plan!!!!")
            print(f"Distance to goal is: {dist_to_goal()}")  
    
    if time_counter == timeout:
        print("Timeout!")


def clb_move_base_done(state, result):
    global status_move_base

    status_move_base = state
    print("Status callback: " + str(state))

def clb_move_base_feedback(feedback):
    #print(feedback)
    pass

def clb_robot_position(msg):
    # Update odometry reading
    global odomm

    current_time = rospy.Time.now()
    odomm.header.stamp = current_time
    odomm.pose.pose.position.x = msg.pose.pose.position.x
    odomm.pose.pose.position.y = msg.pose.pose.position.y
    odomm.pose.pose.orientation.x = msg.pose.pose.orientation.x
    odomm.pose.pose.orientation.y = msg.pose.pose.orientation.y
    odomm.pose.pose.orientation.z = msg.pose.pose.orientation.z
    odomm.pose.pose.orientation.w = msg.pose.pose.orientation.w
    odomm.twist.twist.linear.x = msg.twist.twist.linear.x
    odomm.twist.twist.linear.y = msg.twist.twist.linear.y
    odomm.twist.twist.linear.z = msg.twist.twist.linear.z

    #print(odomm.pose.pose.position)

def clb_viz(msg):
    global vizz

    for i in range(num_tracks):
        vizz.markers[i].pose.position.x = msg.markers[i].pose.position.x
        vizz.markers[i].pose.position.y = msg.markers[i].pose.position.y
    #print(vizz)

def hit_obstacle():
    # Get minimum reading of scanner
    global scann
    global min_dist_to_obstacle

    #print(min(scann.ranges))
    if min(scann.ranges) < min_dist_to_obstacle:
        return True
    return False

def goal_reached():
    # Check if robot reached destination
    global min_dist_to_goal
    global move_base

    if move_base:
        return status_move_base == 3
    else:
        return dist_to_goal() < min_dist_to_goal

def dist_to_goal():
    # Return euclidean distance to goal
    global goal
    global offset # THere seems to be an offset between the robot odom and the move_base pose.
    global odomm

    return np.sqrt((odomm.pose.pose.position.x - goal.position.x + offset[0])**2 + (odomm.pose.pose.position.y - goal.position.y + offset[1])**2)

def handler(signum, frame):
    print("Timeout 20 seconds")
    raise Exception("end of time")

def publish_once(pub, goal):
    # Publish goal pose for robot
    try:
        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            connections = pub.pub.get_num_connections()
            rospy.loginfo('Connections: %d', connections)
            if connections > 0:
                pub.publish(goal)
                print('Published goal')
                rospy.loginfo('Published goal')
                break
            rate.sleep()
    except rospy.ROSInterruptException:
        print('Failed to publish goal')
        raise rospy.loginfo('Failed to publish goal')

def run():
    rospy.init_node('simulate_move_base')
    rospy.Subscriber("scan", LaserScan, clb_scan)
    rospy.Subscriber("odom", Odometry, clb_robot_position)
    rospy.Subscriber("viz", MarkerArray, clb_viz)
    for i in range(0,repetitions):
        # p = multiprocessing.Process(target=run_test, name="run_test")
        # p.start()

        # p.join(40)

        # # If thread is active
        # if p.is_alive():
        #     print("run_test is running... let's kill it...")

        #     # Terminate foo
        #     p.terminate()
        #     p.join()
        run_test()

if __name__ == "__main__":
    run()