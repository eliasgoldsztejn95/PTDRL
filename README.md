# PTDRL 
https://ieeexplore.ieee.org/abstract/document/10342140

https://user-images.githubusercontent.com/75029654/229076291-23a9fd2d-67bf-4096-99ed-5c15fba5552e.mp4

# Requirements

 - ROS 1
 - Python version at least 3.6
 - Python packages: pytorch, rospy

## Running PTDRL on a real robot

PTDRL can be used in combination with move_base:

 - Download ptdrl_robot folder to the robot computer
 - Use move_base package for navigation
 - Run ptdrl.py

For other navigation systems, one needs to change task_env.py, and robot_env.py to connect with the costmaps and with the parameter tuning module.

## Training PTDRL on simulation
- Create your simulation environment of choice. In our case we used a realistic hospital world taken from: https://github.com/aws-robotics/aws-robomaker-hospital-world. We integrated moving people using: https://github.com/srl-freiburg/pedsim_ros. The steps for combining them can be found in **simulation** folder.
- Download your robot. In our case we experimented with Turtlebot and Blattoidea robots.
- Clone **ptdrl** folder. 
- Train using: train_ddqn.py. Test using: test_ddqn.py.

## Simulation, robot and training algorithm
- **ptdrl** was written in the OpenAI ROS fashion. This means that training, simulation, and robot environments are separated.
- For PTDRL, we used DDQN with the train_ddqn.py script. One can use any RL algorithm by replacing: env = task_env.PtdrlTaskEnv() in the training script.
- task_env.py provides all the context we want the robot to learn for the RL task, in this case, navigating fast and safely. It contains the main functions: **step** **reset** and **_init_env_variables**. It includes the move_base related functions: **_set_init_pose**, **_send_goal**, and **tune_parameters**.
- robot_en.py contains all the functions associated to the specific robot that you want to train. It also provides the connection with move_base. It contains the main callback functions: **_scan_callback**, **_odom_callback**, and **_costmap_callback**. 
- params.yaml includes:
1. Which local planner is used
2. The set of parameters of the local planner
3. A list of rooms the robot goes through inside the environment

# Hospital world
<img src="https://user-images.githubusercontent.com/75029654/166143327-e4caf24c-6b8a-4629-9f03-982de54fe37e.png" width="300" height="300">

The simulator of the hospital was taken from: https://github.com/aws-robotics/aws-robomaker-hospital-world.
which is amazon's representation of a hospital. It is very rich in the sense of quantity and quality of objects simulated, and it represents 
realistically a hospital.

### Notes
The models have to be downloaded manually from: https://app.ignitionrobotics.org/fuel/models into local models folder.

# Pedestrain simulator
<img src="https://user-images.githubusercontent.com/75029654/166143081-f978b80b-680e-4c15-87a3-a95c89352896.png" width="500" height="250">

The pedestrian simulation is acquired using pedsim_ros package. The package is based on: https://arxiv.org/pdf/cond-mat/9805244.pdf social force model.
This package allows to choose the quantity and type of pedestrians, and motion patterns. The package was taken from: https://github.com/srl-freiburg/pedsim_ros. Download
and install the package.

### Notes
The hospital world of amazon has to be moved to pedsim_gazebo_plugin/worlds folder. Notice that the line: plugin name="ActorPosesPlugin" filename="libActorPosesPlugin.so"
has to be added at the end of the file to allow pedestrian movement.
  
Notice that the pedestrian simulator has to account for obstacles in the world. This should be described in <scenario>.xml found in pedsim_simulator/secnarios.
