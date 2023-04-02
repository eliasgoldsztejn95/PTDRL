#!/usr/bin/env python3

################################################
## Remove NAN's and publish to filtered_cloud ##
################################################

# For use with multiple-object-tracking-lidar


import rospy
import math
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

# Sensors
costmapp = OccupancyGrid()
vell = Twist()
inff = Float32()


# Publishers
pub_costmap = rospy.Publisher('/costmap_sync', OccupancyGrid, queue_size=10)
pub_vel = rospy.Publisher('/vel_sync', Twist, queue_size=10)
pub_inf = rospy.Publisher('/inf_sync', Float32, queue_size=10)


def clb_costmap(msg):
    # Update costmap reading
    global costmapp

    current_time = rospy.Time.now()
    costmapp.header.stamp = current_time
    costmapp.header.frame_id = msg.header.frame_id
    costmapp.info.height = msg.info.height
    costmapp.info.width = msg.info.width
    costmapp.info.origin.position.x = msg.info.origin.position.x
    costmapp.info.origin.position.y = msg.info.origin.position.y
    costmapp.info.origin.orientation.x = msg.info.origin.orientation.x 
    costmapp.info.origin.orientation.y = msg.info.origin.orientation.y 
    costmapp.info.origin.orientation.z = msg.info.origin.orientation.z 
    costmapp.info.origin.orientation.w = msg.info.origin.orientation.w
    costmapp.data = msg.data

def clb_odom(msg):
    global odomm

    # Update odometry reading
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
    odomm.twist.twist.angular.x = msg.twist.twist.angular.x
    odomm.twist.twist.angular.y = msg.twist.twist.angular.y
    odomm.twist.twist.angular.z = msg.twist.twist.angular.z

def clb_vel(msg):
    global vell

    # Update twist reading
    vell.linear.x = msg.linear.x
    vell.angular.z = msg.angular.z

def clb_inf(msg):
    global inff

    # Update inflation
    inff.data = msg.data

def pub_odom_costmap_synchornized():
    # Change NAN to maximum range and publish to filtered_cloud
    global costmapp
    global vell
    global inff

    pub_costmap.publish(costmapp)
    pub_vel.publish(vell)
    pub_inf.publish(inff)

def run():
    rospy.init_node('odom_costmap')
    print("Publishing odom and costmap synchronized")
    rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, clb_costmap)
    rospy.Subscriber("/pedbot/control/cmd_vel", Twist, clb_vel)
    rospy.Subscriber("/inflation", Float32, clb_inf)
    #rospy.Subscriber("/camera/depth/points", PointCloud2, clb_cloud)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pub_odom_costmap_synchornized()
        rate.sleep()

    rospy.spin()

if __name__ == "__main__":
    run()