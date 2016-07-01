from receive.loadmat import Loadmat
from receive.loadmat_kodama import LoadmatKodama
from receive.udp import UDP
from classifier.svm import SVM
from classifier.libsvm import LibSVM
from classifier.linearsvm import LinearSVM
from classifier.swlinearsvm import StepwiseLinearSVM
from classifier.lda import LDA
from classifier.swlda import SWLDA
import convert.erp

import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=str, help='')
parser.add_argument('session', action='store', type=str, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=15, type=int, help='')
parser.add_argument('--average', dest='average', action='store', default=1, type=int, help='')
parser.add_argument('--skip', dest='skip', action='store', default=0, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--decimate', dest='decimate', action='store', type=int, default=1, help='')
parser.add_argument('--method', dest='method', action='store', type=str, default="l", help='')
parser.add_argument('--log', dest='log', action='store', default='tmp', help='')
parser.add_argument('--no-undersampling', dest='undersampling', action='store_const', const=False, default=True, help='')
parser.add_argument('--undersampling-far', dest='undersampling_far', action='store', type=int, default=0, help='')
parser.add_argument('--undersampling-method', dest='undersampling_method', action='store', type=str, default="cosine", help='')
parser.add_argument('--filename', dest='filename', action='store', type=str, default="../mat/512hz4555/sub%s_sec%s.mat", help='')
parser.add_argument('--matfile', dest='matfile', action='store', type=str, default=None, help='')
parser.add_argument('--modelname', dest='modelname', action='store', type=str, default="tmp", help='')
parser.add_argument('--kodama', dest='kodama', action='store_const', const=True, default=False, help='')
args = parser.parse_args()

print("Subject: %s  Session: %s" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
skip_num = args.skip
block_num = pattern_num * repetition_num

if args.online:
    receiver = UDP(args.subject, args.session, "train", average=args.average, logname=args.log)
elif args.kodama:
    receiver = LoadmatKodama(args.subject, args.session, "train")
else:
    receiver = Loadmat(args.subject, args.session, "train", average=args.average, filename=args.filename, matfile=args.matfile)

for i in range(pattern_num):
    for _ in range(block_num):
        receiver.receive()
    if skip_num != 0:
        for _ in range(skip_num * pattern_num):
            receiver.receive()
        receiver.clear()

receiver.group()

erps = receiver.fetch()

if args.undersampling and args.method != "swlda":
    erps = convert.erp.undersampling(erps, block_num, method=args.undersampling_method, far=args.undersampling_far)
    # erps = convert.erp.undersampling(erps, block_num, method="euclidean", far=60)

labels = sum([list(np.repeat(i, len(erps[i]))) for i in range(len(erps))], [])
erps = sum(erps, [])

if args.method == "rbf":
    classifier = SVM(name=args.modelname, decimate=args.decimate)
elif args.method == "linear" or args.method == "l":
    classifier = LinearSVM(name=args.modelname, decimate=args.decimate)
elif args.method == "swlinearsvm":
    classifier = StepwiseLinearSVM(name=args.modelname, decimate=args.decimate)
elif args.method == "libsvm":
    classifier = LibSVM(name=args.modelname)
elif args.method == "lda":
    classifier = LDA(name=args.modelname)
elif args.method == "swlda":
    classifier = SWLDA(name=args.modelname, decimate=args.decimate)

classifier.train(labels, erps)
receiver.save()
