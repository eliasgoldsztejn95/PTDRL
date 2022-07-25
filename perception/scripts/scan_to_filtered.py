#!/usr/bin/env python3

################################################
## Remove NAN's and publish to filtered_cloud ##
################################################

# For use with multiple-object-tracking-lidar


import rospy
import math
from sensor_msgs.msg import LaserScan, PointCloud2
import laser_geometry.laser_geometry as lg

# Sensors
scann = LaserScan()
cloud = PointCloud2()

# Pointcloud
lp = lg.LaserProjection()

# Publishers
pub_cloud = rospy.Publisher('/filtered_cloud', PointCloud2, queue_size=10)
max = 15

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

def clb_cloud(msg):
    # Update lidar reading
    global cloud

    current_time = rospy.Time.now()
    cloud.header.stamp = current_time
    cloud.header.frame_id = msg.header.frame_id 
    cloud.height = msg.height
    cloud.width = msg.width
    cloud.fields = msg.fields
    cloud.is_bigendian = msg.is_bigendian
    cloud.point_step = msg.point_step
    cloud.row_step = msg.row_step
    cloud.data = msg.data

def remove_nan_cloud():
    # Change NAN to maximum range and publish to filtered_cloud
    global pub_cloud

    filtered_cloud = cloud
    data = list(filtered_cloud.data)

    for i in range(len(data)):
        if math.isnan(data[i]):
            data[i] = max
    #filtered_cloud.data = bytes(data)
    #print(filtered_cloud)
    
    pub_cloud.publish(filtered_cloud)

def remove_nan_scan():
    # Change NAN to maximum range and publish to filtered_cloud
    global pub_cloud
    global fields

    filtered_scann = scann
    ranges = list(filtered_scann.ranges)

    for i in range(len(ranges)):
        if math.isinf(ranges[i]):
            ranges[i] = max
    filtered_scann.ranges = tuple(ranges)

    pc2_msg = lp.projectLaser(filtered_scann)
    
    pub_cloud.publish(pc2_msg)



def run():
    rospy.init_node('scan_to_filter')
    print("Publishing without NAN's")
    rospy.Subscriber("scan", LaserScan, clb_scan)
    #rospy.Subscriber("/camera/depth/points", PointCloud2, clb_cloud)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        remove_nan_scan()
        rate.sleep()

    rospy.spin()

if __name__ == "__main__":
    run()