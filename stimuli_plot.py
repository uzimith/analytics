from receive.loadmat import Loadmat
from receive.udp import UDP

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('session', action='store', type=int, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=15, type=int, help='')
parser.add_argument('--channel', dest='channel_num', action='store', type=int, default=8, help='')
parser.add_argument('--block', dest='block', action='store', type=int, default=1, help='')
args = parser.parse_args()

print("Subject: %d  Session: %d" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
channel_num = args.channel_num
block_num = pattern_num * repetition_num

receiver = Loadmat(args.subject, args.session, "predict", separate=True)

for i in range(pattern_num * block_num):
    receiver.receive()

receiver.group()
erps_of_stimuli = receiver.fetch()

frame_length = len(erps_of_stimuli[0][0][0])
x_units = [int(x) for x in np.linspace(0, frame_length, num=5)]
x_labels = ["0", "200", "400", "600", "800"]

for i in range(channel_num):
    plt.subplot(4, 2, i + 1)
    block = args.block
    erps = [erps[(block - 1) * repetition_num:block * repetition_num] for erps in erps_of_stimuli]

    for erp in erps[0]:
        plt.plot(erp[i], color=[1, 0, 0, 0.2])
    for erp in erps[1]:
        plt.plot(erp[i], color=[0, 1, 0, 0.2])
    for erp in erps[2]:
        plt.plot(erp[i], color=[0, 0, 1, 0.2])
    for erp in erps[3]:
        plt.plot(erp[i], color=[1, 1, 0, 0.2])
    for erp in erps[4]:
        plt.plot(erp[i], color=[1, 0, 1, 0.2])
    for erp in erps[5]:
        plt.plot(erp[i], color=[0, 1, 1, 0.2])

    plt.xlabel("time [ms]")
    plt.ylabel("mu volt")
    plt.xlim([0,frame_length])
    plt.xticks(x_units, x_labels)
plt.show()

print("plotted\n")
