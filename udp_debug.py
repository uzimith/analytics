from loadmat import Loadmat
from udp import UDP
from mysvm import SVM
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('session', action='store', type=int, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=15, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--method', dest='method', action='store', type=str, default="svm", help='')
args = parser.parse_args()

print("Subject: %d  Session: %d" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
block_num = pattern_num * repetition_num

if args.online:
    receiver = UDP()
else:
    receiver = Loadmat(args.subject, args.session, "train")

for i in range(pattern_num * block_num):
    receiver.receive()
receiver.group()

erps = receiver.fetch()

erps[0] = random.sample(erps[0], block_num)

labels = sum([list(np.repeat(i, len(erps[i]))) for i in range(len(erps))], [])
erps = sum(erps, [])

if args.method == "svm":
    svm = SVM()
    svm.train(labels, erps)
print("training is finished\n")
