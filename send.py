import socket

from struct import *

import numpy as np
import scipy.io
from itertools import chain

import argparse
import time

parser = argparse.ArgumentParser(description='training ERP')
parser.add_argument('ip', action='store', type=str, help='')
parser.add_argument('subject', action='store', type=str, help='')
parser.add_argument('train', action='store', type=str, help='')
parser.add_argument('type', action='store', type=str, help='')
args = parser.parse_args()

chunnel_num = 8
# addr = "localhost"
addr = args.ip
port = "8081"
addr_pair = (addr, int(port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mat = scipy.io.loadmat("../mat_files/subject%s_section%s.mat" % (args.subject, args.train))

if args.type == "train":
    labels = mat['target_label']
else:
    labels = mat['stimuli_label']


for (erp, label) in zip(mat['erps'], labels):
    data = list(chain.from_iterable(erp))
    data.append(label)
    packed_data = ""
    for d in data:
        packed_data += pack('f', d)
    sock.sendto(packed_data, addr_pair)
    print(label)

print(len(labels))
