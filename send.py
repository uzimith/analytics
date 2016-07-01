import convert.erp

import socket
from struct import *
import scipy.io
import numpy as np

import argparse
import time

parser = argparse.ArgumentParser(description='training ERP')
parser.add_argument('ip', action='store', type=str, help='')
parser.add_argument('subject', action='store', type=int, help='')
parser.add_argument('train', action='store', type=int, help='')
parser.add_argument('type', action='store', type=str, help='')
parser.add_argument('--filename', dest='filename', action='store', type=str, default="../mat/512hz4555/sub%s_sec%d.mat", help='')
parser.add_argument('--decimate', dest='decimate', action='store', type=int, default=1, help='')
args = parser.parse_args()

chunnel_num = 8
# addr = "localhost"
addr = args.ip
port = "8081"
addr_pair = (addr, int(port))
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mat = scipy.io.loadmat(args.filename % (args.subject, args.train))

if args.type == "train":
    labels = mat['target_label']
else:
    labels = mat['stimuli_label']

erps = mat['erps']
erps = convert.erp.decimate(erps, args.decimate)

for (erp, label) in zip(erps, labels):
    data = erp
    data.append(label)
    packed_data = ""
    for d in data:
        packed_data += pack('f', d)
    sock.sendto(packed_data, addr_pair)
    print(label)
    time.sleep(0.01)

print(len(labels))
