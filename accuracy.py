from loadmat import Loadmat
from udp import UDP
from mysvm import SVM
from mylinearsvm import LinearSVM
import numpy as np
import random
import argparse
import csv

parser = argparse.ArgumentParser(description='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('session', action='store', type=int, help='')
parser.add_argument('--repeat', dest='repeat', action='store', default=5, type=int, help='')
parser.add_argument('--online', dest='online', action='store_const', const=True, default=False, help='')
parser.add_argument('--method', dest='method', action='store', type=str, default="svm", help='')
parser.add_argument('--tmp', dest='tmp', action='store', default='tmp', help='')
args = parser.parse_args()

print("Subject: %d  Session: %d" % (args.subject, args.session))

pattern_num = 6
repetition_num = args.repeat
block_num = pattern_num * repetition_num
success_count = 0
say_count = 0

if args.online:
    receiver = UDP("predict")
else:
    receiver = Loadmat(args.subject, args.session, "predict")

if args.method == "svm":
    classifier = SVM()
    classifier.load()
if args.method == "linear":
    classifier = LinearSVM()
    classifier.load()

for answer in range(1, 6 + 1):
    for i in range(block_num):
        receiver.receive()
    receiver.group()

    erps = receiver.fetch()

    labels = sum([list(np.repeat(i, len(erps[i]))) for i in range(len(erps))], [])
    erps = sum(erps, [])
    result = classifier.predict(labels, erps, pattern_num)
    print("answer: %d reuslt: %d" % (answer, result))
    if(answer == result):
        success_count = success_count + 1
    say_count = say_count + 1


accuracy = 100.0 * success_count / say_count
print("\naccuracy: %f %% (%d)\n" % (accuracy , success_count ) )

print("predicting is finished\n")

with open('log/%s.csv' % args.tmp, 'a') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow([args.subject, args.session, success_count, accuracy])
