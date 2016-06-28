from receive import Receive
import socket
from struct import *
import numpy as np
from operator import itemgetter
from itertools import groupby

class UDP(Receive):
    def __init__(self, type, port = 8081, channel_num=8, average=1):
        Receive.__init__(self, channel_num=8, average=average)
        self.host = socket.gethostbyname(socket.gethostname())
        self.port = port
        self.bind()
        self.type = type

    def bind(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        print("udp : ", (self.host, self.port))

    def receive(self, skip = False):
        a = self.sock.recv(65535)
        if skip:
            return
        data = []
        for i in range(0, len(a), 4):
            tmp = unpack('f', a[i:i+4])
            data.append(tmp[0])
        label = int(data.pop())
        erp = np.asarray(data)
        self.labels.append(label)
        self.erps.append(erp)

    def save(self, filename="tmp"):
        sio.savemat(file_name, {'erps': erps, 'labels': label})
        print("saved: %s" % file_name)

    def train(self, filename, class_weight = False):
        self.class_weight = class_weight
        print("start training")
        print("number of trials : %s" % self.trial_num)
        print("class_weight : %d" % class_weight)
        averaged_num = self.trial_num / self.average_count
        data_num = self.trial_num / self.pattern_num / self.average_count

        self.repetition_num = 5
