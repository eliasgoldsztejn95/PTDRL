#!/usr/bin/env python3

###########################################
## Start test for navigation performance ##
###########################################

# When called, this script moves the robot in
# a straight line at random speeds, for a
# certain amount of time.

import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose, Twist
import numpy as np


#speeds = [0.5, 0.75, 1, 1.25, 1.5]
speeds = [0.5]
num = 30
pub_vel = rospy.Publisher('/pedbot/control/cmd_vel', Twist, queue_size=10)
model_name = "turtlebot3_waffle"

def run():
    rospy.init_node('simulate')
    print("new")
    rate = rospy.Rate(10) # 10hz
    vel = Twist()
    init_pose = Pose()
    init_pose.position.x = 5
    init_pose.position.y = 5
    init_pose.position.z = 0
    orientation = quaternion_from_euler(0,0,-np.pi/2)
    
    init_pose.orientation.x =  orientation[0]
    init_pose.orientation.y =  orientation[1]
    init_pose.orientation.z =  orientation[2]
    init_pose.orientation.w =  orientation[3]

    for i in range(num):
        for speed in speeds:

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

            print("Setting" + model_name + "result: " + str(result))

            vel.linear.x = speed # set speed
            vel.linear.y = speed # set speed
            vel.linear.z = speed # set speed
            for h in range(int(300/speed)):
                pub_vel.publish(vel) # publish speed
                rate.sleep()


if __name__ == "__main__":
    run()