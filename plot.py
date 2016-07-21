import convert.erp
from receive.loadmat import Loadmat
from receive.loadmat_kodama import LoadmatKodama
from receive.loadmat_multi import LoadmatMulti
from receive.udp import UDP
import data

import numpy as np
from scipy import stats
import scipy.io
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('session', action='store', type=int, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=15, type=int, help='')
parser.add_argument('--average', dest='average', action='store', default=1, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--decimate', dest='decimate', action='store', type=int, default=1, help='')
parser.add_argument('--type', dest='type', action='store', type=str, default="mean", help='')
parser.add_argument('--undersampling', dest='undersampling', action='store_const', const=True, default=False, help='')
parser.add_argument('--undersampling-method', dest='undersampling_method', action='store', type=str, default="cosine", help='')
parser.add_argument('--undersampling-far', dest='undersampling_far', action='store', type=int, default=0, help='')
parser.add_argument('--channel', dest='channel_num', action='store', type=int, default=8, help='')
parser.add_argument('--block', dest='block', action='store', type=int, default=1, help='')
parser.add_argument('--filename', dest='filename', action='store', type=str, default="../mat/512hz4555/sub%s_sec%d.mat", help='')
parser.add_argument('--matfile', dest='matfile', action='store', type=str, default=None, help='')
parser.add_argument('--multi', dest='multi', action='store', type=str, default=None, help='')
parser.add_argument('--kodama', dest='kodama', action='store', type=str, default=None, help='')
args = parser.parse_args()

print("Subject: %d  Session: %d" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
channel_num = args.channel_num
block_num = pattern_num * repetition_num

if args.multi:
    receiver = LoadmatMulti(args.multi, "train")
elif args.online:
    receiver = UDP(args.subject, args.session, "plot", average=args.average, logname=args.log)
elif args.kodama:
    receiver = LoadmatKodama(args.subject, args.session, "train", folder=args.kodama, repetition_num=repetition_num)
else:
    receiver = Loadmat(args.subject, args.session, "train", average=args.average, filename=args.filename, matfile=args.matfile)

for i in range(pattern_num * block_num):
    receiver.receive()

receiver.group()

erps = receiver.fetch()

# erps = [convert.erp.highpass(erp) for erp in erps]
# erps = [convert.erp.lowpass(erp) for erp in erps]

if args.undersampling:
    erps = convert.erp.undersampling(erps, block_num, method=args.undersampling_method, far=args.undersampling_far)
if args.decimate != 1:
    erps[0] = convert.erp.decimate(erps[0], args.decimate)
    erps[1] = convert.erp.decimate(erps[1], args.decimate)

# import pdb; pdb.set_trace()

target_data = erps[1]
non_target_data = erps[0]

target_data     = convert.erp.separate(target_data)
non_target_data = convert.erp.separate(non_target_data)

frame_length = len(target_data[0][0]) - 1
x_units = [int(x) for x in np.linspace(0, frame_length, num=5)]
x_labels = ["0", "200", "400", "600", "800"]

font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size'   : 20}
plt.rc('font', **font)
plt.figure(figsize=(30,30))

if args.type == "all":
    for i in range(channel_num):
        plt.subplot(4, 2, i + 1)
        plt.title(data.electrodes()[i])
        plt.grid()
        labeled = False
        for erp in non_target_data:
            if labeled:
                plt.plot(erp[i], color=[0, 0, 1, 0.3])
            else:
                plt.plot(erp[i], color=[0, 0, 1, 0.3], label="non-target")
                labeled = True
        labeled = False
        for erp in target_data:
            if labeled:
                plt.plot(erp[i], color=[1, 0, 0, 0.3])
            else:
                plt.plot(erp[i], color=[1, 0, 0, 0.3], label="target")
                labeled = True
        plt.xlabel("time [ms]")
        plt.ylabel("[uV ]")
        plt.xlim([0,frame_length])
        plt.xticks(x_units, x_labels)
        if i == 1:
            plt.legend(loc = 'upper right')

if args.type == "block":
    block = args.block
    target_data = target_data[(block - 1) * repetition_num:block * repetition_num]
    non_target_data = non_target_data[(block - 1) * (pattern_num - 1) * repetition_num:block * (pattern_num - 1) * repetition_num]
    for i in range(channel_num):
        plt.subplot(4, 2, i + 1)
        plt.title(data.electrodes()[i])
        plt.grid()
        labeled = False
        for erp in non_target_data:
            if labeled:
                plt.plot(erp[i], color=[0, 0, 1, 0.3])
            else:
                plt.plot(erp[i], color=[0, 0, 1, 0.3], label="non-target")
                labeled = True
        labeled = False
        for erp in target_data:
            if labeled:
                plt.plot(erp[i], color=[1, 0, 0, 0.3])
            else:
                plt.plot(erp[i], color=[1, 0, 0, 0.3], label="target")
                labeled = True
        plt.xlabel("time [ms]")
        plt.ylabel("[uV ]")
        plt.xlim([0,frame_length])
        plt.xticks(x_units, x_labels)
        if i == 1:
            plt.legend(loc = 'upper right')

if args.type == "mean":
    average_target_data = np.mean(target_data, axis=0)
    std_target = stats.sem(target_data, axis=0)
    average_non_target_data = np.mean(non_target_data, axis=0)
    std_non_target = stats.sem(non_target_data, axis=0)

    for i in range(channel_num):
        plt.subplot(4, 2, i + 1)
        plt.title(data.electrodes()[i])
        plt.grid()
        plt.errorbar(range(len(average_non_target_data[i])), average_non_target_data[i], yerr=std_non_target[i], color=[0, 0, 1, 0.7], label="non-target")
        plt.errorbar(range(len(average_target_data[i])), average_target_data[i], yerr=std_target[i], color=[1, 0, 0, 0.7], label="target")
        plt.xlabel("time [ms]")
        plt.ylabel("[uV ]")
        plt.xlim([0,frame_length])
        plt.xticks(x_units, x_labels)
        if i == 1:
            plt.legend(loc = 'upper right')

    scipy.io.savemat("log/plot", {
        'target_erp': average_target_data,
        'target_standard_error': std_target,
        'non_target_erp': average_non_target_data,
        'non_target_standard_error': std_non_target
    })

if args.type == "tmp":
    average_target_data = np.mean(target_data, axis=0)
    std_target = stats.sem(target_data, axis=0)
    average_non_target_data = np.mean(non_target_data, axis=0)
    std_non_target = stats.sem(non_target_data, axis=0)

    for i in [2]:
        plt.grid()
        plt.errorbar(range(len(average_non_target_data[i])), average_non_target_data[i], yerr=std_non_target[i], color=[0, 0, 1, 0.7], label="non-target", lw=3)
        plt.errorbar(range(len(average_target_data[i])), average_target_data[i], yerr=std_target[i], color=[1, 0, 0, 0.7], label="target", lw=3)
        plt.xlabel("time [ms]")
        plt.ylabel("[uV ]")
        plt.xlim([0,frame_length])
        plt.xticks(x_units, x_labels)
        plt.legend(loc = 'upper right')

    scipy.io.savemat("log/plot", {
        'target_erp': average_target_data,
        'target_standard_error': std_target,
        'non_target_erp': average_non_target_data,
        'non_target_standard_error': std_non_target
    })

if args.type == "erp":
    plt.figure(figsize=(30,15))
    font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size'   : 0}
    plt.rc('font', **font)
    for j in range(10):
        plt.plot(convert.erp.combine(non_target_data[j]), color=[0, 0, 0, 1], lw=5)
        plt.grid()
        plt.xlabel("")
        plt.xlim([0,frame_length])
        plt.ylim([-20,20])
        print(frame_length)
        plt.savefig('log/feature_non_target%d.png'% (j))
        plt.clf()

if args.multi:
    plt.savefig('log/grand%d%d.png'% (args.subject, args.session))
    print('saved: log/grand%d%d.png'% (args.subject, args.session))

else:
    plt.savefig('log/sub%d-ses%d.png'% (args.subject, args.session))
    print('saved: log/sub%d-ses%d.png'% (args.subject, args.session))
plt.show()


print("plotted\n")
