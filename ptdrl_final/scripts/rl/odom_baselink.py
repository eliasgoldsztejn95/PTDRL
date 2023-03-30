#!/usr/bin/env python
import rospy 
import tf 
from nav_msgs.msg import Odometry 

def odom_callback(msg):     
  br = tf.TransformBroadcaster()     
  br.sendTransform((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
  msg.pose.pose.orientation.z, msg.pose.pose.orientation.w),rospy.Time.now(),"base_footprint","odom") 


if __name__ == '__main__':     
  rospy.init_node('odom_to_base_link_tf')     
  rospy.Subscriber('odom', Odometry, odom_callback)     
  rospy.spin()
