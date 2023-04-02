#!/usr/bin/env python3
import numpy as np 
import os 
import yaml
from yaml.loader import SafeLoader

import matplotlib.pyplot as plt

import matplotlib.animation as animation
import matplotlib.cm as cm

###############################################################
# Open costmap.yaml and transform it to numpy. Costmap is 60x60
###############################################################

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.split("/")
dir_path = '/'.join(dir_path)

print(dir_path)

file = "yaml/costmap_sync_2"
yaml_file = "/" + file + ".yaml"
np_file = "/" + file + ".npy"

dir_path_yaml = dir_path
dir_path_np = dir_path

dir_path_yaml += yaml_file
dir_path_np += np_file

costmap_size = 60

def main():

    with open(dir_path_yaml, 'r') as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))

    seq_len = len(data)
    print(seq_len)
    sequence = np.zeros([seq_len,costmap_size,costmap_size])

    itr = 0
    seq_itr = 0
    while itr < seq_len - 1:
        elapsed_time = 1#int(data[itr]["header"]["stamp"]["nsecs"]/100000000)
        for jtr in range(elapsed_time):
            sequence[seq_itr,:,:] = np.array_split(data[itr]["data"], costmap_size)
            print(data[itr]["header"]["seq"])
            if itr < seq_len - 2:
                if int(data[itr]["header"]["seq"]) != int(data[itr+1]["header"]["seq"]) - 1:
                    print("here")
                    print(data[itr]["header"]["seq"])
                    print(data[itr+1]["header"]["seq"])
            seq_itr += 1
        itr += 1
    print(seq_itr)

    f.close()

    with open(dir_path_np, 'wb') as f:
        np.save(f, sequence)
    f.close()

    with open(dir_path_np, 'rb') as f:
       a = np.load(f)


    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(2000):
        frames.append([plt.imshow(a[i,:,:], cmap=cm.Greys_r,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                    repeat_delay=1)

    plt.show()



if __name__ == '__main__':
    main()