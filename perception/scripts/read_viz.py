#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import MarkerArray
import numpy as np 
import os 
import yaml
from yaml.loader import SafeLoader

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.split("/")
dir_path = dir_path[:-2]
dir_path += ["recordings"]
dir_path = '/'.join(dir_path)

file = "tracking2"
yaml_file = "/" + file + ".yaml"
np_file = "/" + file + ".npy"

dir_path_yaml = dir_path
dir_path_np = dir_path

dir_path_yaml += yaml_file
dir_path_np += np_file

# Parameters
num_clusters = 6
recording_duration = 10.8 # in minutes
seq_len = 3 # in ~0.5 seconds

#sequence = np.zeros([num_clusters*2, seq_len*2, int((recording_duration*600)/(seq_len))])
sequence = np.zeros([int((recording_duration*600)/(seq_len)), seq_len*2, num_clusters*2])
print(sequence.shape)

def main():

    with open(dir_path_yaml, 'r') as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))
    print(len(data))
    for seq in range(int((recording_duration*600)/(seq_len))):
         for step in range(seq_len*2):
             for cluster in range(num_clusters):
                #sequence[cluster*2,step,seq] = data[seq*(seq_len) + step]["markers"][cluster]["pose"]["position"]["x"]
                #sequence[cluster*2+1,step,seq] = data[seq*(seq_len) + step]["markers"][cluster]["pose"]["position"]["y"]
                 sequence[seq,step,cluster*2] = data[seq*(seq_len) + step]["markers"][cluster]["pose"]["position"]["x"]
                 sequence[seq,step,cluster*2+1] = data[seq*(seq_len) + step]["markers"][cluster]["pose"]["position"]["y"]
    f.close()

    with open(dir_path_np, 'wb') as f:
        np.save(f, sequence)
    f.close()

    with open(dir_path_np, 'rb') as f:
        a = np.load(f)

    print(data[11]["markers"][0])


    print(a[0,11,0:2])
    print(a[1,1,0:2])

    print(a[195,11,0:2])
    print(a[196,1,0:2])

if __name__ == '__main__':
    main()