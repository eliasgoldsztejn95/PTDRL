# PTDRL 
This repository includes all file to work with PTDRL: Parameter Tuning with Deep Reinforcement Learning

ptdrl_final contains all trained networks, and by running ptdrl.py one can run our code in a robot using move_base.


https://user-images.githubusercontent.com/75029654/224107735-2a7a3fe1-47c2-44c8-9ddf-bdc33c6c4283.mp4



# Hospital_bot
Hospital simulator with pedestrians and robot

This repository includes files and commands used for the creation of a hospital simulator with pedestrians and a robot.

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
This package allows to choose the quantity and type of pedestrians, and motion patterns. THe package was taken from: https://github.com/srl-freiburg/pedsim_ros.

### Notes
The hospital world of amazon has to be moved to pedsim_gazebo_plugin/worlds folder. Notice that the line: plugin name="ActorPosesPlugin" filename="libActorPosesPlugin.so"
has to be added at the end of the file to allow pedestrian movement.
  
Notice that the pedestrian simulator has to account for obstacles in the world. This should be described in <scenario>.xml found in pedsim_simulator/secnarios.
  
  
### Commands
To launch the hospital world:
> roslaunch pedsim_gazebo_plugin hospital.launch

To launch the pedestrian simulator:
> roslaunch pedsim_simulator simple_pedestrians.launch

### Commands
To launch the available robots launch:
> roslaunch blattoidea blattoidea.launch

> roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
