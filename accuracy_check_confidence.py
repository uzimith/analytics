from receive.loadmat import Loadmat
from receive.loadmat_kodama import LoadmatKodama
from receive.udp import UDP
from classifier.svm import SVM
from classifier.libsvm import LibSVM
from classifier.linearsvm import LinearSVM
from classifier.swlinearsvm import StepwiseLinearSVM
from classifier.lda import LDA
from classifier.swlda import SWLDA

import socket
import numpy as np
import random
import argparse
import csv
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=str, help='')
parser.add_argument('session', action='store', type=str, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=5, type=int, help='')
parser.add_argument('--average', dest='average', action='store', default=1, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--decimate', dest='decimate', action='store', type=int, default=1, help='')
parser.add_argument('--method', dest='method', action='store', type=str, default="l", help='')
parser.add_argument('--log', dest='log', action='store', default='tmp', help='')
parser.add_argument('--problem', dest='problem', action='store', default=1, type=int, help='')
parser.add_argument('--skip', dest='skip', action='store', default=0, type=int, help='')
parser.add_argument('--filename', dest='filename', action='store', type=str, default="../mat/512hz4555/sub%s_sec%s.mat", help='')
parser.add_argument('--matfile', dest='matfile', action='store', type=str, default=None, help='')
parser.add_argument('--modelname', dest='modelname', action='store', type=str, default="tmp", help='')
parser.add_argument('--kodama', dest='kodama', action='store', type=str, default=None, help='')
parser.add_argument('--no-say-result', dest='say_result', action='store_const', const=False, default=True, help='')
parser.add_argument('--send-udp', dest='send_udp', action='store_const', const=True, default=False, help='')
parser.add_argument('--ip', dest='ip', action='store', type=str, default="172.0.0.10", help='')
parser.add_argument('--port', dest='port', action='store', type=int, default=12347, help='')
args = parser.parse_args()

# print("Subject: %s  Session: %s" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
problem_num = args.problem
skip_num = args.skip
block_num = pattern_num * repetition_num
success_count = 0
say_count = 0

if args.online:
    receiver = UDP(args.subject, args.session, "predict", average=args.average, logname=args.log)
elif args.kodama:
    receiver = LoadmatKodama(args.subject, args.session, "predict", folder=args.kodama, repetition_num=repetition_num)
else:
    receiver = Loadmat(args.subject, args.session, "predict", average=args.average, filename=args.filename, matfile=args.matfile)

if args.method == "rbf":
    classifier = SVM(name=args.modelname, decimate=args.decimate)
elif args.method == "libsvm":
    classifier = LibSVM(name=args.modelname)
elif args.method == "linear" or args.method == "l":
    classifier = LinearSVM(name=args.modelname, decimate=args.decimate)
elif args.method == "swlinearsvm":
    classifier = StepwiseLinearSVM(name=args.modelname, decimate=args.decimate)
elif args.method == "lda":
    classifier = LDA(name=args.modelname)
elif args.method == "swlda":
    classifier = SWLDA(name=args.modelname, decimate=args.decimate)

classifier.load()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with open('log/%s.csv' % args.log, 'a') as f:
    for answer in range(1, pattern_num + 1):
        for _ in range(problem_num):
            for i in range(block_num):
                receiver.receive()

            receiver.group()
            erps = receiver.fetch()

            labels = sum([list(np.repeat(i, len(erps[i]))) for i in range(len(erps))], [])
            erps = sum(erps, [])
            (result, summary) = classifier.predict_proba(labels, erps, pattern_num)
            summary.sort(reverse=True)
            writer = csv.writer(f, lineterminator='\n')
            f.write("%s, %s, %d, %d, %d, %f\n" % (args.subject, args.session, answer, result, answer == result, summary[0] - summary[1]))

        for _ in range(skip_num * pattern_num):
            receiver.receive(skip=True)


receiver.save()
