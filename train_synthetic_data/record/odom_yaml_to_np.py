#!/usr/bin/env python3

import numpy as np 
import os 
import yaml
from yaml.loader import SafeLoader

import matplotlib.pyplot as plt


from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
xdata = np.linspace(0,2,2)
ydata = [0]*2
ln, = ax.plot([], [])
a = []


###############################################################
# Open <environments>.yaml and transform it to numpy.
###############################################################

# open yaml file
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.split("/")
dir_path = '/'.join(dir_path)

print(dir_path)

file = "yaml/odom_sync"
yaml_file = "/" + file + ".yaml"
np_file = "/" + file + ".npy"

dir_path_yaml = dir_path
dir_path_np = dir_path

dir_path_yaml += yaml_file
dir_path_np += np_file



def main():
    global a

    with open(dir_path_yaml, 'r') as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))

    seq_len = len(data)
    print(seq_len)
    sequence = np.zeros([seq_len, 2])

    for i in range(seq_len -1):
        sequence[i,0] = data[i]["twist"]["twist"]["linear"]["x"]
        sequence[i,1] = data[i]["twist"]["twist"]["angular"]["z"]
        if i < seq_len - 2:
            if int(data[i]["header"]["seq"]) != int(data[i+1]["header"]["seq"]) - 1:
                print("here")
                print(data[i]["header"]["seq"])
                print(data[i+1]["header"]["seq"])
    

    with open(dir_path_np, 'wb') as f:
       np.save(f, sequence)
    f.close()

    with open(dir_path_np, 'rb') as f:
       a = np.load(f)

    #for i in range(seq_len -1):
    #   plt.plot(np.linspace(0,720,720), a[i])
    #   plt.show()
    seq_len = len(a)

    indexes=np.linspace(0, seq_len-2, seq_len-1)
    indexes = [int(x) for x in indexes]

    ani = FuncAnimation(fig, update, init_func=init, frames=indexes, blit=True)
    plt.show()


def init():
    ax.set_xlim(0, 2)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    global a
    ydata = a[frame]
    ln.set_data(xdata, ydata)
    return ln,


if __name__ == '__main__':
    main()