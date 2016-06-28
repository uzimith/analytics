from receive.loadmat import Loadmat
from receive.udp import UDP
from classifier.svm import SVM
from classifier.linearsvm import LinearSVM
from classifier.lda import LDA
from classifier.swlda import SWLDA

import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('session', action='store', type=int, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=15, type=int, help='')
parser.add_argument('--average', dest='average', action='store', default=1, type=int, help='')
parser.add_argument('--skip', dest='skip', action='store', default=0, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--normalize', dest='normalize', action='store_const', const=True, default=False, help='')
parser.add_argument('--method', dest='method', action='store', type=str, default="l", help='')
parser.add_argument('--no-undersampling', dest='undersampling', action='store_const', const=False, default=True, help='')
parser.add_argument('--undersampling-far', dest='undersampling_far', action='store', type=int, default=60, help='')
args = parser.parse_args()

print("Subject: %d  Session: %d" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
skip_num = args.skip
block_num = pattern_num * repetition_num

if args.online:
    receiver = UDP("train", average=args.average)
else:
    receiver = Loadmat(args.subject, args.session, "train", normalize=args.normalize, average=args.average)

for i in range(pattern_num):
    for _ in range(block_num):
        receiver.receive()
    if skip_num != 0:
        for _ in range(skip_num * pattern_num):
            receiver.receive()
        receiver.clear()

receiver.group()

if args.undersampling and args.method != "swlda":
    receiver.undersampling(block_num, far=args.undersampling_far)

erps = receiver.fetch()

labels = sum([list(np.repeat(i, len(erps[i]))) for i in range(len(erps))], [])
erps = sum(erps, [])

if args.method == "linear" or args.method == "l":
    classifier = LinearSVM()
if args.method == "svm":
    classifier = SVM()
if args.method == "lda":
    classifier = LDA()
if args.method == "swlda":
    classifier = SWLDA()

classifier.train(labels, erps)

print("training is finished\n")
