#!/usr/bin/env python3
"""
Created on Mon Dec  2 17:03:34 2019

@author: mahmoud
"""

import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import *
from rospkg import RosPack
from pedsim_msgs.msg  import AgentStates

# xml file containing a gazebo model to represent agent, currently is represented by a cubic but can be changed
global xml_file

# list of different actor models. 1 is a normal person, 2 is a wheelchair. It has to be the same size as actor.agents_states.
global model_list
model_list = [1,3,2,3,2,1,2,1,3,1,2]
#model_list = [1,3,2,3,2,1,2]
#model_list = [4,4,4,4,4,4]

def actor_poses_callback(actors):

    rospack1 = RosPack()
    pkg_path = rospack1.get_path('pedsim_gazebo_plugin')
    default_actor_model_file = pkg_path + "/models/actor_model"

    itr = 0
    for actor in actors.agent_states:
        print(model_list[itr])
        actor_model_file = rospy.get_param('~actor_model_file', default_actor_model_file + "_" + str(model_list[itr]) + ".sdf")
        file_xml = open(actor_model_file)
        xml_string = file_xml.read()

        actor_id = str( actor.id )
        actor_pose = actor.pose
        rospy.loginfo("Spawning model: actor_id = %s", actor_id)

        model_pose = Pose(Point(x= actor_pose.position.x,
                               y= actor_pose.position.y,
                               z= actor_pose.position.z),
                         Quaternion(actor_pose.orientation.x,
                                    actor_pose.orientation.y,
                                    actor_pose.orientation.z,
                                    actor_pose.orientation.w) )

        spawn_model(actor_id, xml_string, "", model_pose, "world")
        itr += 1
        
    rospy.signal_shutdown("all agents have been spawned !")




if __name__ == '__main__':

    rospy.init_node("spawn_pedsim_agents")

    print("Waiting for gazebo services...")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    print("service: spawn_sdf_model is available ....")
    rospy.Subscriber("/pedsim_simulator/simulated_agents", AgentStates, actor_poses_callback)

    rospy.spin()
