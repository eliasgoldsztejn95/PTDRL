#!/usr/bin/env python3

################################################
## Remove NAN's and publish to filtered_cloud ##
################################################

# For use with multiple-object-tracking-lidar


import rospy
import math
from sensor_msgs.msg import LaserScan, PointCloud2
import laser_geometry.laser_geometry as lg
from nav_msgs.msg import Odometry, OccupancyGrid

# Sensors
scann = LaserScan()
odomm = Odometry()


# Publishers
pub_scan = rospy.Publisher('/scan_sync', LaserScan, queue_size=10)
pub_odom = rospy.Publisher('/odom_sync', Odometry, queue_size=10)


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

def pub_odom_scan_synchornized():
    # Change NAN to maximum range and publish to filtered_cloud
    global odomm
    global scann

    pub_scan.publish(scann)
    pub_odom.publish(odomm)
    


def run():
    rospy.init_node('odom_scan')
    print("Publishing odom and scan synchronized")
    rospy.Subscriber("scan", LaserScan, clb_scan)
    rospy.Subscriber("odom", Odometry, clb_odom)
    #rospy.Subscriber("/camera/depth/points", PointCloud2, clb_cloud)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        pub_odom_scan_synchornized()
        rate.sleep()

    rospy.spin()

if __name__ == "__main__":
    run()