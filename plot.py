from receive.loadmat import Loadmat
from receive.udp import UDP

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('session', action='store', type=int, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=15, type=int, help='')
parser.add_argument('--average', dest='average', action='store', default=1, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--type', dest='type', action='store', type=str, default="mean", help='')
parser.add_argument('--undersampling', dest='undersampling', action='store_const', const=True, default=False, help='')
parser.add_argument('--normalize', dest='normalize', action='store_const', const=True, default=False, help='')
parser.add_argument('--channel', dest='channel_num', action='store', type=int, default=8, help='')
parser.add_argument('--block', dest='block', action='store', type=int, default=1, help='')
args = parser.parse_args()

print("Subject: %d  Session: %d" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
channel_num = args.channel_num
block_num = pattern_num * repetition_num

if args.online:
    receiver = UDP("plot")
else:
    receiver = Loadmat(args.subject, args.session, "train", separate=True, normalize=args.normalize, average=args.average)

for i in range(pattern_num * block_num):
    receiver.receive()

receiver.group()

if args.undersampling:
    receiver.undersampling(block_num)

erps = receiver.fetch()

target_data = erps[1]
non_target_data = erps[0]

frame_length = len(erps[0][0][0])
x_units = [int(x) for x in np.linspace(0, frame_length, num=5)]
x_labels = ["0", "200", "400", "600", "800"]

if args.type == "all":
    for i in range(channel_num):
        plt.subplot(4, 2, i + 1)
        for erp in target_data:
            plt.plot(erp[i], color=[1, 0, 0, 0.1])
        for erp in non_target_data:
            plt.plot(erp[i], color=[0, 0, 1, 0.1])
        plt.xlabel("time [ms]")
        plt.ylabel("mu volt")
        plt.xlim([0,frame_length])
        plt.xticks(x_units, x_labels)
    plt.show()

if args.type == "block":
    block = args.block
    target_data = target_data[(block - 1) * repetition_num:block * repetition_num]
    non_target_data = non_target_data[(block - 1) * (pattern_num - 1) * repetition_num:block * (pattern_num - 1) * repetition_num]
    for i in range(channel_num):
        plt.subplot(4, 2, i + 1)
        for erp in target_data:
            plt.plot(erp[i], color=[1, 0, 0, 0.1])
        for erp in non_target_data:
            plt.plot(erp[i], color=[0, 0, 1, 0.1])
        plt.xlim([0,frame_length])
    plt.show()

if args.type == "mean":
    average_target_data = np.mean(target_data, axis=0)
    std_target = stats.sem(target_data, axis=0)
    average_non_target_data = np.mean(non_target_data, axis=0)
    std_non_target = stats.sem(non_target_data, axis=0)

    for i in range(channel_num):
        plt.subplot(4, 2, i + 1)
        plt.errorbar(range(len(average_target_data[i])), average_target_data[i], yerr=std_target[i], color=[1, 0, 0, 0.7])
        plt.errorbar(range(len(average_non_target_data[i])), average_non_target_data[i], yerr=std_non_target[i], color=[0, 0, 1, 0.7])
        plt.xlabel("time [ms]")
        plt.ylabel("volt")
        plt.xlim([0,frame_length])
        plt.xticks(x_units, x_labels)

    plt.show()

print("plotted\n")
